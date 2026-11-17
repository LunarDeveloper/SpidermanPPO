"""PPO rollout/update wall times and cumulative simulation/transport counters."""
import json
from pathlib import Path
import time

import torch
from stable_baselines3 import PPO


class ProfiledPPO(PPO):
    def collect_rollouts(self, *args, **kwargs):
        started = time.perf_counter()
        result = super().collect_rollouts(*args, **kwargs)
        self._collection_seconds = time.perf_counter() - started
        return result

    def train(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        super().train()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        update_s = time.perf_counter() - started
        self._update_seconds = getattr(self, "_update_seconds", 0.0) + update_s
        collection_s = getattr(self, "_collection_seconds", 0.0)
        self._total_collection_seconds = getattr(self, "_total_collection_seconds", 0.0) + collection_s
        env = self.get_env().unwrapped
        counters = env.profile()
        record = {"policy_steps": self.num_timesteps, "collection_s": collection_s,
                  "update_s": update_s, "total_collection_s": self._total_collection_seconds,
                  "total_update_s": self._update_seconds, **counters}
        record["session_policy_steps"] = counters.get("policy_steps", 0)
        record["policy_steps"] = self.num_timesteps
        elapsed = max(self._total_collection_seconds + self._update_seconds, 1e-9)
        record["physics_steps_per_s"] = counters.get("physics_steps", 0) / elapsed
        self.logger.record("perf/collection_s", collection_s)
        self.logger.record("perf/update_s", update_s)
        self.logger.record("perf/physics_steps_per_s", record["physics_steps_per_s"])
        # Append after each actual update, including the final update (SB3's
        # regular console dump happens before train()). No device sync per step.
        folder = Path(getattr(self, "profile_dir", self.logger.dir))
        with (folder / "performance.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        # SB3 normally dumps before the update; explicitly flush here so the
        # final update's losses and timing also reach TensorBoard and console.
        self.logger.dump(step=self.num_timesteps)
