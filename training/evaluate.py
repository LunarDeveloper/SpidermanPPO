"""Watch a newly trained Godot policy, with its matching normalization state."""
import argparse
from pathlib import Path
import shutil
import json


def checkpoint_files(run, checkpoint=None):
    if checkpoint is None:
        return run / "final.zip", run / "normalization.pkl"
    prefix, steps, marker = checkpoint.stem.rsplit("_", 2)
    if marker != "steps" or not steps.isdigit():
        raise ValueError("Expected an SB3 checkpoint named PREFIX_NUMBER_steps.zip")
    return checkpoint, checkpoint.with_name(f"{prefix}_vecnormalize_{steps}_steps.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--godot", default=shutil.which("godot") or shutil.which("Godot_v4.7.2-stable_win64_console.exe"))
    parser.add_argument("--port", type=int, default=12000)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--checkpoint", type=Path, help="Evaluate a periodic checkpoint instead of final.zip")
    args = parser.parse_args()
    checkpoint, normalization = checkpoint_files(args.run, args.checkpoint)
    for path in (checkpoint, normalization):
        if not path.is_file():
            parser.error(f"Saved file not found: {path}")
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from godot_env import GodotSpiderEnv
    from batch_env import GodotBatchVecEnv
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; CPU fallback is disabled.")
    config_path = args.run / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    if config.get("backend") == "batched-v2":
        base = GodotBatchVecEnv(args.godot, 1, 1, args.port, config["action_repeat"],
                               config["tick_gamma"], render=True, log_dir=str(args.run / "viewer"))
    else:
        base = DummyVecEnv([lambda: GodotSpiderEnv(args.godot, args.port, True)])
    env = base
    try:
        env = VecNormalize.load(normalization, base)
        env.training = False
        env.norm_reward = False
        model = PPO.load(checkpoint, env=env, device="cuda")
        observation = env.reset()
        count = 0
        total = 0.0
        while count < args.episodes:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, info = env.step(action)
            total += float(info[0].get("raw_reward", reward[0]))
            if done[0]:
                count += 1
                print(f"Episode {count}: return={total:.3f}, distance={info[0]['distance']:.1f}m")
                total = 0.0
    finally:
        env.close()


if __name__ == "__main__":
    main()
