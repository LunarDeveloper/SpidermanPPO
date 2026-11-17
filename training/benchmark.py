"""Headless collector benchmark. Excludes startup and neural-network updates."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--godot", default=shutil.which("godot") or shutil.which("Godot_v4.7.2-stable_win64_console.exe"))
    parser.add_argument("--backend", choices=("legacy", "batched"), default="batched")
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--action-repeat", type=int, choices=(1, 4, 5), default=4)
    parser.add_argument("--batches", type=int, default=256)
    parser.add_argument("--port", type=int, default=16200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.backend == "legacy":
        from stable_baselines3.common.vec_env import SubprocVecEnv
        from godot_env import make_env
        env = SubprocVecEnv([make_env(args.godot, args.port + i, False, "runs/benchmark")
                             for i in range(args.envs)], start_method="spawn")
        repeat = 1
    else:
        from batch_env import GodotBatchVecEnv
        env = GodotBatchVecEnv(args.godot, args.envs, args.workers, args.port,
                               args.action_repeat, log_dir="runs/benchmark")
        repeat = args.action_repeat
    try:
        env.reset()
        rng = np.random.default_rng(42)
        for _ in range(16):
            env.step(rng.uniform(-1, 1, (args.envs, 6)).astype(np.float32))
        before = env.profile() if hasattr(env, "profile") else {}
        started = time.perf_counter()
        ticks = 0
        for _ in range(args.batches):
            _, _, _, infos = env.step(rng.uniform(-1, 1, (args.envs, 6)).astype(np.float32))
            ticks += sum(info.get("physics_steps", 1) for info in infos)
        elapsed = time.perf_counter() - started
        counters = env.profile() if hasattr(env, "profile") else {}
        counters = {k: v - before.get(k, 0) for k, v in counters.items()}
        result = {"backend": args.backend, "envs": args.envs,
                  "workers": args.workers if args.backend == "batched" else args.envs,
                  "action_repeat": repeat, "elapsed_s": elapsed, "physics_steps": ticks,
                  "physics_steps_per_s": ticks / elapsed,
                  "policy_steps_per_s": args.batches * args.envs / elapsed,
                  "json_codec_fraction": ((counters.get("encode_s", 0) + counters.get("decode_s", 0)) / elapsed
                                          if counters else None),
                  "profile": counters}
        print(json.dumps(result, indent=2))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    finally:
        env.close()


if __name__ == "__main__":
    main()
