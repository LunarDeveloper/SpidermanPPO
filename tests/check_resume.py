"""Verify a resumed run continued counters, optimization, and normalization."""
import argparse
import json
from pathlib import Path
import pickle

import torch
from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("continued", type=Path)
    args = parser.parse_args()
    before = PPO.load(args.source / "final.zip", device="cuda")
    after = PPO.load(args.continued / "final.zip", device="cuda")
    config = json.loads((args.continued / "config.json").read_text(encoding="utf-8"))
    assert config["start_policy_steps"] == before.num_timesteps
    assert after.num_timesteps >= before.num_timesteps + config["steps"]
    assert after._n_updates > before._n_updates
    before_steps = [float(state["step"].item()) for state in before.policy.optimizer.state.values()]
    after_steps = [float(state["step"].item()) for state in after.policy.optimizer.state.values()]
    assert min(after_steps) > min(before_steps), "Optimizer counter was reset"
    assert any(not torch.equal(a, b) for a, b in zip(before.policy.parameters(), after.policy.parameters()))
    # Only use trusted test-run pickle files, as with SB3 normalization loading.
    with (args.source / "normalization.pkl").open("rb") as stream:
        old_normalization = pickle.load(stream)
    with (args.continued / "normalization.pkl").open("rb") as stream:
        new_normalization = pickle.load(stream)
    added = after.num_timesteps - before.num_timesteps
    assert new_normalization.obs_rms.count >= old_normalization.obs_rms.count + added
    assert after.gamma == before.gamma == new_normalization.gamma
    print(f"PASS: {before.num_timesteps} -> {after.num_timesteps} steps; optimizer, updated weights, and normalization retained")


if __name__ == "__main__":
    main()
