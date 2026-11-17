# Spider PPO: Godot conversion

The standalone Godot 4 prototype lives in `godot/`. The original Unity project
and its pretrained networks are retained for reference. The new training path
initializes fresh policies by default, can resume its own saved runs, and never
reads `PPO Networks/`.

## Current implementation

- Procedural 11-body ragdoll, pin joints, damped torque-based arm aiming, and two
  retractable tension-only spring webs.
- Continuous street corridor, follow camera, web visuals, and manual sandbox.
- Batched arenas with separate `World3D` physics spaces: eight arenas in two Godot
  processes by default, coordinated by one Python process.
- 100 Hz physics and a default 25 Hz policy (four held physics ticks per action).
  Simulation pauses while waiting for the trainer, so GPU updates do not advance it.
- Batched loopback JSON with 20 observations and 6 continuous actions per arena,
  discounted training rewards, raw episode scores, and independent autoresets.
- Separate PyTorch/Stable Baselines3 PPO trainer using **CUDA explicitly** for
  policy inference and optimization. Missing CUDA is an error, never a CPU fallback.
- Parallel Godot processes batch policy inference, observation/reward normalization,
  periodic checkpoints, TensorBoard logs, and evaluation of newly trained policies.
- Physics-only headless arenas omit meshes, cameras, lights, and visual updates.
- Collection, update, simulator, and host/transport timing counters for profiling.

This is an initial functional training environment, not visual or physics parity
with the Unity scene. The original character mesh, constrained anatomical joints,
traffic, city decoration, sound, and cinematic cameras have not been ported.
Godot physics, networking, rollout storage, and normalization still use CPU resources.
Small MLP policies can be slower on GPU; benchmark environment count and batch size
before committing to long training runs.

## Open the sandbox

Import `godot/project.godot` in Godot 4 (the standard build is sufficient), then run.
Point the **mouse** at a wall. A green crosshair and dashed guides show a valid
attachment point; red means there is no wall target within range. Hold the **left
mouse button / Q** for the left web and **right mouse button / E** for the right
web. Release to detach. Existing webs keep their anchors when you move the cursor,
so you can aim the second web independently. **Space** pauses to line up a shot,
**T** toggles quarter-speed practice, and **R** resets.

Manual shots follow the cursor directly, while arm joints rotate toward the target.
Training still casts along the simulated forearm directions using the original
six-action interface. This manual aiming assistance does not train or load a policy.

## Install GPU training dependencies (Windows / NVIDIA)

Python 3.11+ and a CUDA-capable NVIDIA GPU/driver are required. Python 3.13 is
supported by the selected package range. From the repository root:

```powershell
./scripts/setup-training.ps1
```

The script creates `.venv`, installs a CUDA 13.0 PyTorch wheel and the training
requirements, and verifies GPU access. PyTorch supplies its CUDA runtime;
a separate CUDA toolkit is not required. For a different GPU/driver setup, choose
the appropriate wheel from [PyTorch's installation guide](https://pytorch.org/get-started/locally/).

## Train from scratch

```powershell
./.venv/Scripts/python.exe training/train.py --envs 8 --workers 2 --steps 250000 --output runs/efficient-1
```

If Godot is not on PATH, append `--godot 'C:/path/to/Godot.exe'`.
Training is headless by default. `--render` still offers a first-arena debugging
window, but a separate checkpoint viewer is preferred for throughput. Each invocation
creates a new run folder; networks start randomly unless `--resume` is provided.

To continue a completed or Ctrl+C-saved run for **another** 250000 policy steps:

```powershell
./.venv/Scripts/python.exe training/train.py --resume runs/efficient-2 --steps 250000 --port 17000 --output runs/efficient-2-continued
```

Resume loads `final.zip`, including optimizer state and cumulative step count, plus
`normalization.pkl`. Environment count, rollout size, batch size, seed, and action
repeat default to the source run's configuration. Worker count can change; changing
action repeat is rejected. Physics episodes restart, while learned parameters and
normalization statistics continue. Additional steps round up to a complete rollout.
The new folder preserves the original run and receives new checkpoints, logs, and
final files; future evaluation should use the new folder. Performance counters reset
for the new session while policy-step counters remain cumulative.

Start with a short integration run:

```powershell
./.venv/Scripts/python.exe training/train.py --envs 4 --workers 1 --steps 128 --rollout-steps 32 --batch-size 64 --output runs/gpu-smoke
```

`--steps` counts **policy transitions**, summed across arenas. At the default
action repeat of four, 250000 policy transitions advance approximately one million
physics ticks. Early terminals can use fewer than four ticks. SB3's `time/fps` is
also policy transitions/second; use `perf/physics_steps_per_s` for physical throughput.
Episode lengths are policy decisions; episode `physics_steps` records exact ticks.

Output folders must not already exist, to prevent accidental overwrites. SB3 rounds
requested steps up to a whole rollout (`envs * rollout-steps`). Use Ctrl+C to save
the current model and normalization state and stop. Periodic checkpoint ZIPs have
matching `spider_vecnormalize_NUMBER_steps.pkl` files; the final model uses `normalization.pkl`.
Checkpoints occur approximately every 65536 aggregate physics ticks; filenames use
SB3's **policy-step** count (typically 16384, 32768, etc. with repeat four).
Use `--checkpoint-ticks` to change that approximate physical-tick interval.
Ports default to 11000 through 11000 + worker count - 1 and bind loopback only.
Use `--port` to run independent jobs. Close workers through the trainer; closing a
Godot window while training intentionally fails the connection.

```powershell
./.venv/Scripts/python.exe -m tensorboard.main --logdir runs
./.venv/Scripts/python.exe training/evaluate.py runs/YOUR-RUN
```

Evaluation loads only a selected new run's `final.zip` and matching normalization
statistics; inference also requires CUDA. Only load checkpoint files you trust.

To inspect progress **while training continues**, use a periodic checkpoint and
its matching normalization file in a second terminal. For example, after the
16384-policy-step checkpoint has been saved:

```powershell
./.venv/Scripts/python.exe training/evaluate.py runs/efficient-1 --checkpoint runs/efficient-1/checkpoints/spider_16384_steps.zip --port 15000 --episodes 5
```

The viewer launches its own single-arena Godot process. It reads the run's saved
configuration to match action repeat and discounting. Older runs without the
batched-backend marker continue to use the original 100 Hz adapter. The viewer
loads a snapshot; it does not follow weights as training updates them. Both
training and viewing share hardware, so evaluation can temporarily reduce throughput.

## Control timing and performance

Use `--action-repeat 4` (default) for 25 Hz decisions, `5` for 20 Hz, or `1` for
the original 100 Hz. Each held action is reapplied every physics tick so arm torque,
spring forces, retraction, cooldowns, ground contact, and rewards keep updating at
100 Hz. An arena stops at its terminal tick; its remaining held ticks are skipped,
then it resets independently before the next decision. Other worlds continue.

For repeat `k`, training reward is `r0 + 0.99*r1 + ... + 0.99^(k-1)*r[k-1]`.
PPO and reward normalization use `gamma = 0.99^k`; GAE uses `lambda = 0.95^k`
to keep its decay comparable in physical time. Console `ep_rew_mean` remains the
**undiscounted** episode score, with the same movement/attachment/ground rewards.
The default rollout length is about ten simulated seconds per arena (256 decisions
at 25 Hz). The policy now has fewer opportunities to change actions, so its control
problem is intentionally different even though the time-based discount is preserved.
Start a fresh run; old policies can load but were trained at a different control rate.

`--workers` selects Godot process count (default two, or one for a single arena).
Arena counts need not divide evenly across workers. Each arena has its own physics
space and local coordinates, rather than relying on large distances for isolation.
There are no per-arena Python subprocesses in the new training path.

Each PPO update appends `performance.jsonl` in the run folder:

- `collection_s` / `total_collection_s`: rollout wall time, including inference,
  simulator waiting, and rollout bookkeeping.
- `update_s` / `total_update_s`: PPO update wall time, synchronized with CUDA at
  update boundaries. There is no added per-step CUDA synchronization for profiling.
- `env_wall_s`: cumulative batched step send/wait time; `server_critical_path_s`
  sums the slowest worker's reported simulation time for each batch.
- `host_and_transport_s`: nonnegative residual between those two timings, including
  codecs, sockets, polling and scheduling. This is **not** pure network latency.
- `encode_s`, `decode_s`, `send_s`, `receive_wait_s`, and byte counts: client codec
  and socket measurements. Receive waiting overlaps simulation, so these counters
  must not all be added together. `server_script_s` sums controller/transition code
  time across workers and excludes engine physics and frame-boundary waiting.
- `physics_steps_per_s`: actual physics ticks divided by measured collection and
  update time, excluding startup. `simulator-profile.json` saves cumulative counters
  at the end, and performance summaries also appear in the console/TensorBoard.

Collector-only benchmarks (exclude startup, inference, and optimization):

```powershell
./.venv/Scripts/python.exe training/benchmark.py --backend legacy --envs 8 --batches 128 --port 16200
./.venv/Scripts/python.exe training/benchmark.py --envs 8 --workers 2 --action-repeat 1 --batches 128 --port 16200
./.venv/Scripts/python.exe training/benchmark.py --envs 8 --workers 2 --action-repeat 4 --batches 128 --port 16200
```

Initial short measurements on this machine:

| Collector | Physics ticks/second |
| --- | ---: |
| Original eight-process path, 100 Hz policy | 3756 |
| Eight batched arenas / two processes, 100 Hz policy | 5321 |
| Eight batched arenas / two processes, 25 Hz policy | 8834 |

These are short local measurements subject to machine load, not a guarantee of
end-to-end training speed or learning improvement. An eight-arena CUDA rollout
measured 1.62 s collecting experience and 0.48 s updating PPO (about 3884 actual
physics ticks/s over those phases). JSON client encoding/decoding was about 5% of
the 25 Hz collector benchmark and about 2% of that CUDA collection/update time.
Batched JSON is retained for now;
binary buffers/shared memory remain a possible next step if profiling a larger run
shows codecs or transport are a significant bottleneck.

## Environment definition

Coordinates use +Y up and +Z progress. Episodes begin at height 55 m and forward
speed 12 m/s, with no webs attached. The corridor is 53 m wide and 120 m high.
All resets currently use this fixed starting state; the seed is accepted for Gym
compatibility and future randomization.

The observation order is:

1. Pelvis X/Y position (2).
2. Pelvis XYZ velocity (3).
3. Left and right forearm outward XYZ directions (6).
4. Left and right attachment flags, -1 or +1 (2).
5. Progress relative to the moving minimum-distance threshold (1).
6. Left and right wall anchors relative to the pelvis, or zero when detached (6).

Actions in [-1, 1] are left yaw/pitch, right yaw/pitch, left cast, right cast.
Yaw spans ±pi and pitch spans ±pi/2. Positive cast attaches; nonpositive releases.
Both the upper arm and forearm rotate toward the requested world-space direction
using inertia-scaled proportional/derivative torque control. Angular damping
reduces overshoot, acceleration and torque are capped, and equal opposing joint
torques act on the parent segment/chest. The controller handles 180-degree turns
without snapping body rotations. Training webs still launch along the **actual**
forearm direction, so aiming takes simulated time; yaw/pitch are target directions.
Webs use
stiffness 200, damping 5, retraction 3 m/s, and maximum attachment distance 140 m.
Web tension is capped at 20000 N as a numerical stability safeguard.

Reward is forward speed^1.02 * 0.25 * height bonus * center bonus * dt.
Shooting, attaching, holding, and releasing webs have no reward cost.
Negative forward speed is clamped before the fractional power.
Anchor observations now describe the wall attachment, rather than the elbow anchor.
Contact between **any ragdoll body part and the street** applies an additional
one-time -0.5 reward and ends the episode. Building-wall contact does not count.
The old 2 m pelvis-height cutoff was removed because it ended episodes before
physical ground contact could occur. Falling below ground level remains a safety
termination, and falling more than 2 m behind the moving threshold still ends an
episode. Episodes truncate after 3000 ticks (30 simulated seconds), preserving
time-limit bootstrapping through Gymnasium/SB3. The positive movement reward
and one-time -0.5 ground-contact penalty remain in effect.

Height is already observation index 1, signed XYZ velocity is at indices 2–4,
and actual left/right unit arm directions are at indices 5–7 and 8–10 (zero based).
Velocity magnitude gives speed, and each unit direction encodes yaw/pitch without
angle wrapping. The observation/action shapes remain 20/6; previous checkpoints
can load, but arm dynamics and rewards have changed. Start a new training run for
a clean comparison. Already-running Godot processes must restart to use changes.
Step `info` also reports height, speed, forward speed, left/right aim error in
degrees, ground-contact status, and the ground penalty separately.

## Verify

```powershell
py tests/smoke_bridge.py
./.venv/Scripts/python.exe tests/smoke_batch.py
./.venv/Scripts/python.exe tests/smoke_batch.py --action-repeat 5
Godot_v4.7.2-stable_win64_console.exe --headless --path godot --script res://tests/manual_aim.gd
Godot_v4.7.2-stable_win64_console.exe --headless --fixed-fps 100 --path godot --script res://tests/arm_control.gd
```

The dependency-free test starts real headless Godot and verifies one-tick stepping,
paused waiting, repeatable resets, finite observations/rewards, episode termination,
and penalty-free attachment geometry/rewards. Simulator logs are written under `runs/`.
The batch tests compare independent-world states and discounted/raw reward sums
against a one-tick reference, then verify real terminal observations, partial held
steps, ground penalties, and independent autoresets at 25 Hz and 20 Hz.
The arm-controller test checks real-physics convergence for forward and backward
targets (about 6–7 degrees error after 1.5 simulated seconds in the test), bounded
180-degree turning torque, actual height/velocity/direction observations, and
street-contact termination with a one-time -0.5 penalty and clean reset.
The short CUDA training command above additionally exercises spawned environments,
batched policy inference, PPO backpropagation, checkpoint saving, and normalization.

Next work is to measure training throughput and learning progress, tune ragdoll
joint limits and rewards, then port the visual scene. A successful smoke test does
not establish that the policy has learned to swing.

Initial validation on this machine: Godot 4.7.2 headless startup and the live bridge
smoke test passed, including three complete random-action episodes. Python modules
passed compilation checks. Fresh CUDA PPO updates and saving of the model and
normalization statistics have been verified on an NVIDIA RTX 3500 Ada GPU, including
a visible environment alongside a headless worker. Godot window visibility is
independent of Gym's render mode, so mixed visible/headless workers can share a
vector environment. These short checks do not establish learned swinging behavior.
On the current Python 3.15 installation, the rendered training checks also reported
a native Windows access violation during process exit, after both checkpoint files
were saved. The cause of that shutdown issue has not yet been isolated; explicitly
closing the TensorBoard writer did not resolve it.

GPU PPO behavior and configuration: [Stable Baselines3 PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).
