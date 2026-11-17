"""Gymnasium adapter for a dedicated Godot simulation process."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import subprocess
import time
import uuid

import gymnasium as gym
import numpy as np

PROJECT = Path(__file__).resolve().parents[1] / "godot"


class GodotSpiderEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 100}

    def __init__(self, godot: str, port: int = 11000, render: bool = False,
                 log_dir: str = "runs/simulator"):
        super().__init__()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (20,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (6,), np.float32)
        # `render` controls Godot's external window, not Gym's render() API.
        # All vector workers expose the same mode, even when one has a window.
        self.render_mode = None
        self.process = None
        self.sock = None
        self.stream = None
        self.owns_peer = False
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_dir) / f"godot-{port}.log"
        self.log = self.log_path.open("w", encoding="utf-8")
        command = [godot, "--path", str(PROJECT)]
        if not render:
            command += ["--headless", "--fixed-fps", "100"]
        token = uuid.uuid4().hex
        command += ["--", "--port", str(port), "--token", token]
        try:
            self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 30
            while True:
                if self.process.poll() is not None:
                    raise RuntimeError(f"Godot exited at startup. See {self.log_path}")
                try:
                    self.sock = socket.create_connection(("127.0.0.1", port), timeout=1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Godot did not start. See {self.log_path}")
                    time.sleep(0.1)
            self.sock.settimeout(30)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.stream = self.sock.makefile("rwb")
            hello = self._request({"command": "hello"})
            if hello != {"protocol": 1, "token": token, "observations": 20, "actions": 6}:
                raise RuntimeError(f"Unexpected simulator on port {port}; choose a different --port")
            self.owns_peer = True
            self.reset()
        except BaseException:
            self.close()
            raise

    def _request(self, payload):
        self.stream.write((json.dumps(payload, allow_nan=False) + "\n").encode())
        self.stream.flush()
        line = self.stream.readline(65537)
        if not line or len(line) > 65536:
            raise ConnectionError(f"Invalid or closed Godot connection; see {self.log_path}")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    @staticmethod
    def _observation(result):
        obs = np.asarray(result["observation"], dtype=np.float32)
        if obs.shape != (20,) or not np.isfinite(obs).all():
            raise ValueError(f"Invalid simulator observation: {obs}")
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        result = self._request({"command": "reset", "seed": seed or 0})
        return self._observation(result), result["info"]

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("Expected six finite actions")
        result = self._request({"command": "step", "action": action.clip(-1, 1).tolist()})
        reward = float(result["reward"])
        if not np.isfinite(reward):
            raise ValueError("Nonfinite simulator reward")
        return (self._observation(result), reward, bool(result["terminated"]),
                bool(result["truncated"]), result["info"])

    def close(self):
        if self.stream is not None and self.owns_peer:
            try:
                # Graceful exit also closes Godot's Windows console-wrapper child.
                self._request({"command": "close"})
            except (OSError, ValueError, RuntimeError):
                pass
        self.owns_peer = False
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        if self.sock is not None:
            self.sock.close()
            self.sock = None
        if self.process is not None:
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process = None
        if self.log is not None:
            self.log.close()


def make_env(godot, port, render, log_dir):
    """Top-level picklable factory for Windows spawn workers."""
    def create():
        from stable_baselines3.common.monitor import Monitor
        return Monitor(GodotSpiderEnv(godot, port, render, log_dir))
    return create
