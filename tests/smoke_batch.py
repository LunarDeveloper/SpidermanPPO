"""Live tests for isolated arenas, action holding, reward discounting and autoreset."""
import argparse
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from batch_env import GodotBatchVecEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--godot", default=shutil.which("godot") or shutil.which("Godot_v4.7.2-stable_win64_console.exe"))
    parser.add_argument("--port", type=int, default=16100)
    parser.add_argument("--action-repeat", type=int, choices=(4, 5), default=4)
    args = parser.parse_args()
    repeat = args.action_repeat
    batch = GodotBatchVecEnv(args.godot, 2, 1, args.port, repeat, log_dir="runs/batch-test")
    reference = None
    try:
        reference = GodotBatchVecEnv(args.godot, 1, 1, args.port + 1, 1, log_dir="runs/batch-test")
        observations = batch.reset()
        reference.reset()
        assert observations.shape == (2, 20)
        assert len(set(batch.workers[0].hello["spaces"])) == 2
        left = np.array([-0.25, 0.4, 0.25, 0.4, 1, 1], np.float32)
        right = np.array([-0.8, -0.3, 0.8, -0.3, -1, -1], np.float32)
        obs, rewards, dones, infos = batch.step(np.stack([left, right]))
        discounted = raw = 0.0
        for k in range(repeat):
            reference_obs, reference_reward, _, reference_info = reference.step(left[None])
            discounted += 0.99 ** k * float(reference_reward[0])
            raw += reference_info[0]["raw_reward"]
        np.testing.assert_allclose(obs[0], reference_obs[0], atol=0.002, rtol=0.0001)
        np.testing.assert_allclose(rewards[0], discounted, atol=1e-5)
        np.testing.assert_allclose(infos[0]["raw_reward"], raw, atol=1e-5)
        assert infos[0]["steps"] == infos[1]["steps"] == repeat
        assert not np.allclose(obs[0], obs[1])
        # Terminals must return the reset observation AND retain the true terminal
        # state separately for SB3, without running through an episode boundary.
        batch.reset()
        action = np.array([[-0.25, 0.4, 0.25, 0.4, -1, -1]] * 2, np.float32)
        saw_terminal = False
        for _ in range(800):
            obs, rewards, dones, infos = batch.step(action)
            assert np.isfinite(obs).all() and np.isfinite(rewards).all()
            if dones.any():
                for i in np.flatnonzero(dones):
                    info = infos[i]
                    assert "episode" in info and "terminal_observation" in info
                    assert info["episode"]["physics_steps"] == info["steps"]
                    assert 1 <= info["physics_steps"] <= repeat
                    np.testing.assert_allclose(obs[i, 1], 55, atol=1e-4)
                    if info["ground_contact"]:
                        assert info["ground_penalty"] == 0.5
                    saw_terminal = True
                break
        assert saw_terminal
        obs, _, dones, infos = batch.step(action)
        assert infos[0]["steps"] == repeat and not dones[0]
        # Malformed action batches must fail before sending anything to Godot.
        try:
            batch.step_async(np.zeros((1, 6)))
            raise AssertionError("Wrong action shape accepted")
        except ValueError:
            pass
        print(f"PASS: isolated worlds, {repeat}-tick holding, discounted/raw rewards, terminal state, autoreset, validation")
    finally:
        if reference is not None:
            reference.close()
        batch.close()


if __name__ == "__main__":
    main()
