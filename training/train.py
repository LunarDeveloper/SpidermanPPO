"""Fresh or resumed CUDA PPO training. Never imports Unity checkpoints."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--godot", default=shutil.which("godot") or shutil.which("Godot_v4.7.2-stable_win64_console.exe"))
    parser.add_argument("--envs", type=int, default=None)
    parser.add_argument("--resume", type=Path, help="Run folder containing final.zip and normalization.pkl")
    parser.add_argument("--steps", type=int, default=1_000_000,
                        help="Total policy transitions across all arenas (each advances up to action-repeat physics ticks)")
    parser.add_argument("--workers", type=int, default=None, help="Godot processes; default min(2, envs)")
    parser.add_argument("--action-repeat", type=int, choices=(1, 4, 5), default=None,
                        help="100 Hz physics ticks per policy decision: 1=100 Hz, 4=25 Hz, 5=20 Hz")
    parser.add_argument("--rollout-steps", type=int, default=None,
                        help="Policy steps per arena per PPO rollout; default preserves about 10 seconds")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--port", type=int, default=11000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-ticks", type=int, default=65536,
                        help="Approximate aggregate physics ticks between automatic checkpoints")
    parser.add_argument("--render", action="store_true", help="Show the first environment")
    parser.add_argument("--output", type=Path, default=Path("runs") / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    source = {}
    if args.resume:
        for name in ("final.zip", "normalization.pkl", "config.json"):
            if not (args.resume / name).is_file():
                parser.error(f"Resume file not found: {args.resume / name}")
        source = json.loads((args.resume / "config.json").read_text(encoding="utf-8"))
        if source.get("backend") != "batched-v2":
            parser.error("Resume currently requires a batched-v2 training run")
    for key, default in (("envs", 8), ("action_repeat", 4), ("batch_size", 512), ("seed", 42)):
        if getattr(args, key) is None:
            setattr(args, key, source.get(key, default))
    if source and args.action_repeat != source["action_repeat"]:
        parser.error("Resuming requires the saved action-repeat to preserve control timing and discounting")
    if args.workers is None:
        args.workers = min(2, args.envs)
    if args.rollout_steps is None:
        args.rollout_steps = source.get("rollout_steps") or max(64, (1024 // args.action_repeat) // 64 * 64)
    if not args.godot:
        parser.error("Godot executable not found. Supply --godot PATH.")
    if args.envs < 1 or args.steps < 1 or args.rollout_steps < 2 or args.batch_size < 2:
        parser.error("Positive environments/steps and rollout/batch sizes >= 2 required")
    if args.checkpoint_ticks < 1:
        parser.error("checkpoint-ticks must be positive")
    if not 1 <= args.workers <= args.envs or args.port < 1024 or args.port + args.workers > 65536:
        parser.error("Require 1 <= workers <= envs and valid nonprivileged ports")
    if args.envs * args.rollout_steps % args.batch_size:
        parser.error("batch-size must divide envs * rollout-steps")

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Install a CUDA-enabled PyTorch build; CPU fallback is disabled.")

    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import VecNormalize
    from batch_env import GodotBatchVecEnv
    from ppo_metrics import ProfiledPPO

    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {**vars(args), "output": str(args.output), "gpu": torch.cuda.get_device_name(0),
                "resume": str(args.resume) if args.resume else None,
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "initialization": "resumed" if args.resume else "random",
                "backend": "batched-v2", "physics_hz": 100, "policy_hz": 100 / args.action_repeat,
                "tick_gamma": 0.99, "gamma": 0.99 ** args.action_repeat,
                "gae_lambda": 0.95 ** args.action_repeat}
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"{'Resuming training' if args.resume else 'Training fresh networks'} on {metadata['gpu']}: {args.envs} arenas in {args.workers} Godot processes, "
          f"100 Hz physics / {metadata['policy_hz']:g} Hz policy", flush=True)
    base = GodotBatchVecEnv(args.godot, args.envs, args.workers, args.port,
                           args.action_repeat, render=args.render, log_dir=str(args.output / "simulator"))
    env = base
    model = None
    try:
        if args.resume:
            env = VecNormalize.load(args.resume / "normalization.pkl", base)
            env.training = True
            env.norm_reward = True
            model = ProfiledPPO.load(args.resume / "final.zip", env=env, device="cuda",
                                     n_steps=args.rollout_steps, batch_size=args.batch_size,
                                     tensorboard_log=str(args.output / "tensorboard"))
            if abs(model.gamma - metadata["gamma"]) > 1e-10 or abs(env.gamma - model.gamma) > 1e-10:
                raise ValueError("Saved model/normalization discount does not match the run configuration")
        else:
            env = VecNormalize(base, norm_obs=True, norm_reward=True, clip_obs=10, gamma=metadata["gamma"])
            model = ProfiledPPO("MlpPolicy", env, device="cuda", seed=args.seed,
                    n_steps=args.rollout_steps, batch_size=args.batch_size, n_epochs=8,
                    learning_rate=3e-4, gamma=metadata["gamma"], gae_lambda=metadata["gae_lambda"], clip_range=0.2,
                    ent_coef=0.005, target_kl=0.01,
                    policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
                    tensorboard_log=str(args.output / "tensorboard"), verbose=1)
        model.profile_dir = str(args.output)
        model._update_seconds = model._total_collection_seconds = model._collection_seconds = 0.0
        metadata["start_policy_steps"] = model.num_timesteps
        metadata["target_policy_steps"] = model.num_timesteps + args.steps
        (args.output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Starting at {model.num_timesteps} policy steps; adding {args.steps} steps "
              "(rounded up to a full rollout).", flush=True)
        assert next(model.policy.parameters()).is_cuda
        callback = CheckpointCallback(save_freq=max(args.checkpoint_ticks // (args.envs * args.action_repeat), 1),
                                      save_path=str(args.output / "checkpoints"),
                                      save_vecnormalize=True, name_prefix="spider")
        try:
            model.learn(total_timesteps=args.steps, callback=callback, reset_num_timesteps=not bool(args.resume))
        except KeyboardInterrupt:
            print("Interrupted; saving current model and normalization statistics.")
        model.save(args.output / "final")
        env.save(args.output / "normalization.pkl")
        # Keep a complete cumulative counter snapshot even for interrupted runs.
        (args.output / "simulator-profile.json").write_text(
            json.dumps(env.unwrapped.profile(), indent=2), encoding="utf-8")
    finally:
        try:
            # Flush TensorBoard and join its writer thread before interpreter exit.
            if model is not None and hasattr(model, "_logger"):
                model.logger.close()
        finally:
            env.close()


if __name__ == "__main__":
    main()
