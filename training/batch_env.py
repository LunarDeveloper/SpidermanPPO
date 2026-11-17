"""SB3 VecEnv: batched Godot arenas, without one Python worker per arena."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import socket
import subprocess
import time
import uuid

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

PROJECT = Path(__file__).resolve().parents[1] / "godot"


class BatchWorker:
    def __init__(self, godot, port, count, repeat, tick_gamma, render, log_dir):
        self.count = count
        self.sock = self.stream = self.process = None
        self.owned = False
        self.metrics = defaultdict(float)
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_dir) / f"batch-{port}.log"
        self.log = self.log_path.open("w", encoding="utf-8")
        token = uuid.uuid4().hex
        command = [godot, "--path", str(PROJECT)]
        if not render:
            command += ["--headless", "--fixed-fps", "100"]
        command += ["res://batch.tscn", "--", "--port", str(port), "--token", token,
                    "--arenas", str(count), "--repeat", str(repeat), "--tick-gamma", str(tick_gamma)]
        try:
            self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT)
            deadline = time.perf_counter() + 30
            while True:
                if self.process.poll() is not None:
                    raise RuntimeError(f"Godot exited: see {self.log_path}")
                try:
                    self.sock = socket.create_connection(("127.0.0.1", port), timeout=1)
                    break
                except OSError:
                    if time.perf_counter() > deadline:
                        raise TimeoutError(f"Godot startup timed out: see {self.log_path}")
                    time.sleep(0.05)
            self.sock.settimeout(30)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.stream = self.sock.makefile("rb")
            self.send({"command": "hello"})
            hello = self.receive()
            if hello.get("protocol") != 2 or hello.get("token") != token:
                raise RuntimeError(f"Unexpected simulator on port {port}; use another --port")
            self.owned = True
            if (hello["arenas"], hello["repeat"], hello["observations"], hello["actions"]) != (count, repeat, 20, 6):
                raise RuntimeError(f"Simulator configuration mismatch: {hello}")
            if len(set(hello["spaces"])) != count:
                raise RuntimeError("Arenas do not have isolated physics worlds")
            self.hello = hello
        except BaseException:
            self.close()
            raise

    def send(self, payload):
        started = time.perf_counter()
        data = (json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n").encode()
        self.metrics["encode_s"] += time.perf_counter() - started
        started = time.perf_counter()
        self.sock.sendall(data)
        self.metrics["send_s"] += time.perf_counter() - started
        self.metrics["sent_bytes"] += len(data)

    def receive(self):
        started = time.perf_counter()
        data = self.stream.readline(1048577)
        self.metrics["receive_wait_s"] += time.perf_counter() - started
        if not data or len(data) > 1048576 or not data.endswith(b"\n"):
            raise ConnectionError(f"Invalid/closed simulator response: see {self.log_path}")
        started = time.perf_counter()
        result = json.loads(data)
        self.metrics["decode_s"] += time.perf_counter() - started
        self.metrics["received_bytes"] += len(data)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def close(self):
        if self.sock is not None:
            # Closing the trainer connection makes protocol-2 Godot exit, even
            # when a rollout was interrupted with a response still outstanding.
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        if self.sock is not None:
            self.sock.close()
            self.sock = None
        if self.process is not None:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process = None
        self.log.close()


class GodotBatchVecEnv(VecEnv):
    def __init__(self, godot, num_envs=8, workers=2, port=11000, action_repeat=4,
                 tick_gamma=0.99, render=False, log_dir="runs/simulator"):
        if not 1 <= workers <= num_envs or action_repeat not in (1, 4, 5):
            raise ValueError("Require 1 <= workers <= num_envs and action_repeat in (1, 4, 5)")
        self.action_repeat = action_repeat
        self.workers = []
        self.pending = False
        self.metrics = defaultdict(float)
        self.started = time.perf_counter()
        self.episode_rewards = np.zeros(num_envs, dtype=np.float64)
        self.episode_lengths = np.zeros(num_envs, dtype=np.int64)
        self.episode_ticks = np.zeros(num_envs, dtype=np.int64)
        try:
            for i in range(workers):
                count = num_envs // workers + (i < num_envs % workers)
                self.workers.append(BatchWorker(godot, port + i, count, action_repeat,
                                                tick_gamma, render and i == 0, log_dir))
            super().__init__(num_envs, gym.spaces.Box(-np.inf, np.inf, (20,), np.float32),
                             gym.spaces.Box(-1, 1, (6,), np.float32))
        except BaseException:
            self.close()
            raise

    def reset(self):
        if self.pending:
            raise RuntimeError("Cannot reset while step_async is pending")
        for worker in self.workers:
            worker.send({"command": "reset"})
        observations = []
        for worker in self.workers:
            observations.extend(worker.receive()["observations"])
        self.episode_rewards.fill(0)
        self.episode_lengths.fill(0)
        self.episode_ticks.fill(0)
        self.reset_infos = [{"physics_dt": 0.01, "action_repeat": self.action_repeat}
                            for _ in range(self.num_envs)]
        self._reset_seeds()
        self._reset_options()
        return self._observations(observations)

    def _observations(self, values):
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (self.num_envs, 20) or not np.isfinite(values).all():
            raise ValueError("Invalid batched observations")
        return values

    def step_async(self, actions):
        if self.pending:
            raise RuntimeError("Step already pending")
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, 6) or not np.isfinite(actions).all():
            raise ValueError("Expected num_envs x 6 finite actions")
        self.step_started = time.perf_counter()
        offset = 0
        self.pending = True
        for worker in self.workers:
            worker.send({"command": "step", "actions": actions[offset:offset + worker.count].clip(-1, 1).tolist()})
            offset += worker.count

    def step_wait(self):
        if not self.pending:
            raise RuntimeError("No pending step")
        results = []
        server_times = []
        for worker in self.workers:
            batch = worker.receive()
            if len(batch["results"]) != worker.count:
                raise ValueError("Simulator result count mismatch")
            results.extend(batch["results"])
            server_times.append(batch["timing"]["batch_s"])
            self.metrics["server_script_s"] += batch["timing"]["script_s"]
        self.pending = False
        env_wall = time.perf_counter() - self.step_started
        server_wall = max(server_times)
        self.metrics["env_wall_s"] += env_wall
        self.metrics["server_critical_path_s"] += server_wall
        # Residual includes codecs, sockets, polling and scheduling; it is an
        # estimate of host/transport overhead, not a pure network latency timer.
        self.metrics["host_and_transport_s"] += max(0, env_wall - server_wall)
        self.metrics["batch_calls"] += 1
        rewards = np.zeros(self.num_envs, np.float32)
        dones = np.zeros(self.num_envs, bool)
        infos = []
        for i, result in enumerate(results):
            rewards[i] = result["reward"]
            dones[i] = result["terminated"] or result["truncated"]
            ticks = int(result["physics_steps"])
            self.metrics["physics_steps"] += ticks
            self.metrics["policy_steps"] += 1
            self.episode_rewards[i] += result["raw_reward"]
            self.episode_lengths[i] += 1
            self.episode_ticks[i] += ticks
            info = result["info"]
            info["raw_reward"] = result["raw_reward"]
            info["physics_steps"] = ticks
            info["TimeLimit.truncated"] = bool(result["truncated"] and not result["terminated"])
            if dones[i]:
                info["terminal_observation"] = np.asarray(result["terminal_observation"], np.float32)
                info["episode"] = {"r": float(self.episode_rewards[i]), "l": int(self.episode_lengths[i]),
                                   "t": time.perf_counter() - self.started,
                                   "physics_steps": int(self.episode_ticks[i]), "distance": info["distance"]}
                self.episode_rewards[i] = self.episode_lengths[i] = self.episode_ticks[i] = 0
            infos.append(info)
        if not np.isfinite(rewards).all():
            raise ValueError("Nonfinite batched reward")
        return self._observations([r["observation"] for r in results]), rewards, dones, infos

    def profile(self):
        values = dict(self.metrics)
        for worker in self.workers:
            for key, value in worker.metrics.items():
                values[key] = values.get(key, 0) + value
        return values

    def get_attr(self, attr_name, indices=None):
        selected = self._get_indices(indices)
        values = {"render_mode": None, "metadata": {"render_modes": []},
                  "action_repeat": self.action_repeat}
        if attr_name not in values:
            raise AttributeError(attr_name)
        return [values[attr_name] for _ in selected]

    def set_attr(self, attr_name, value, indices=None):
        raise AttributeError(f"Simulator configuration is immutable: {attr_name}")

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise AttributeError(f"Unsupported remote method: {method_name}")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]

    def close(self):
        for worker in self.workers:
            worker.close()
        self.workers.clear()
