"""Dependency-free live Godot protocol and physics regression checks."""
import argparse
import json
import math
from pathlib import Path
import random
import shutil
import socket
import subprocess
import time
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--godot", default=shutil.which("godot") or shutil.which("Godot_v4.7.2-stable_win64_console.exe"))
    parser.add_argument("--port", type=int, default=11990)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    log_dir = root / "runs" / "smoke"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "godot.log").open("w", encoding="utf-8") as log:
        token = uuid.uuid4().hex
        process = subprocess.Popen([args.godot, "--headless", "--fixed-fps", "100",
                                    "--path", str(root / "godot"), "--", "--port", str(args.port), "--token", token],
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 20
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Godot exited; inspect runs/smoke/godot.log")
                try:
                    sock = socket.create_connection(("127.0.0.1", args.port), timeout=1)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.1)
            with sock, sock.makefile("rwb") as stream:
                sock.settimeout(10)

                def request(command, **fields):
                    stream.write((json.dumps({"command": command, **fields}) + "\n").encode())
                    stream.flush()
                    result = json.loads(stream.readline())
                    assert "error" not in result, result
                    if "observation" in result:
                        assert len(result["observation"]) == 20
                        assert all(math.isfinite(x) for x in result["observation"])
                    return result

                hello = request("hello")
                assert hello["token"] == token and hello["protocol"] == 1, hello
                baseline = request("reset", seed=42)["observation"]
                assert abs(baseline[1] - 55) < 0.01, baseline
                assert abs(baseline[4] - 12) < 0.01, baseline
                # No requests means no simulated time, independent of wall time.
                time.sleep(0.2)
                result = request("step", action=[0, 0, 0, 0, -1, -1])
                assert result["info"]["steps"] == 1
                assert 0.05 < result["info"]["distance"] < 0.2, result
                assert 54.98 < result["observation"][1] <= 55.01, result
                assert math.isfinite(result["reward"])
                for episode in range(3):
                    reset = request("reset", seed=42)["observation"]
                    assert max(abs(a - b) for a, b in zip(reset, baseline)) < 0.001, reset
                    for step in range(3100):
                        action = [random.uniform(-1, 1) for _ in range(6)]
                        result = request("step", action=action)
                        assert result["info"]["steps"] == step + 1
                        assert math.isfinite(result["reward"])
                        if result["terminated"] or result["truncated"]:
                            break
                    assert result["terminated"] or result["truncated"], result
                    if result["info"]["ground_contact"]:
                        assert result["terminated"] and result["info"]["ground_penalty"] == 0.5, result
                    print(f"Episode {episode}: {step + 1} finite steps, distance={result['info']['distance']:.2f}")
                # Successful attachments must not subtract from movement reward.
                request("reset")
                attached = request("step", action=[-0.5, 0, 0.5, 0, 1, 1])
                assert attached["observation"][11:13] == [1, 1], attached
                obs = attached["observation"]
                expected = max(obs[4], 0) ** 1.02 * 0.25
                expected *= 0.5 + 0.5 * min(max(obs[1] / 120, 0), 1)
                expected *= min(max((53 - abs(obs[0])) / 53, 0.8), 1) * 0.01
                assert abs(attached["reward"] - expected) < 1e-6, attached
                assert abs(attached["observation"][14]) > 20, attached
                request("close")
                print("PASS: lockstep timing, finite observations/rewards, episode reset, penalty-free spring attachment")
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=10)


if __name__ == "__main__":
    main()
