"""
Quantitative robustness eval for a trained Go2 policy (Tier-2 PMTG) -- "watch it and judge" -> numbers.

Identical methodology to the legacy eval (eval_policy_legacy.py): run the DETERMINISTIC policy over N
seeded episodes; each episode reaches a steady gait, then a SCRIPTED external torso shove is applied
(fixed timing + magnitude; direction cycled over the 8 compass points) and recovery is measured:
  * fall rate            -- fraction that terminate (fall) after the shove
  * recovery rate        -- fraction that return to a steady upright stance and hold it
  * mean time-to-upright -- avg seconds from shove-end to sustained-upright (recovered episodes only)
  * mean dip             -- avg lowest torso height reached after the shove
Results are written to eval_<steps>_steps.txt in the checkpoint folder, for the research log.

It loads the env from THIS run's train.py, so the 16-D action / 60-D obs space is handled
automatically -- nothing here is action-space-specific.

Usage:
  python eval_policy.py 4 --dir checkpoints
  python eval_policy.py 0 --dir checkpoints --episodes 32 --force 90
  for F in 40 60 80 100 120; do python eval_policy.py 4 --dir checkpoints --force $F; done   # find the cliff
"""
import os
import argparse
import importlib.util

import numpy as np
import gymnasium as gym
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# ---- Recovery thresholds (what counts as "back on its feet") --------------------------------------
UP_OK = 0.85          # torso_up (cos of tilt) at/above this = roughly upright (~32 deg or less)
H_LO, H_HI = 0.28, 0.40   # torso height band (m) that counts as a proper standing stance
HOLD_OK_S = 0.5       # must hold the upright stance this long (s) to count as recovered

parser = argparse.ArgumentParser(description="Quantitative push-recovery eval for a Go2 policy.")
parser.add_argument("checkpoint", nargs="?", type=int, default=4,
                    help="checkpoint index (1=750k ... 4=3M, 0=latest/Ctrl+C save). Default 4.")
parser.add_argument("--dir", default="checkpoints",
                    help="checkpoint folder (name relative to this script, or absolute). Default: checkpoints.")
parser.add_argument("--episodes", type=int, default=24, help="number of eval episodes (default 24).")
parser.add_argument("--force", type=float, default=75.0, help="scripted shove force in N (default 75).")
parser.add_argument("--dur", type=float, default=0.2, help="scripted shove duration in s (default 0.2).")
parser.add_argument("--settle", type=float, default=2.0, help="settle time before the shove, s (default 2.0).")
parser.add_argument("--window", type=float, default=3.0, help="recovery window after the shove, s (default 3.0).")
parser.add_argument("--seed", type=int, default=0, help="base RNG seed (default 0).")
parser.add_argument("--terrain", action="store_true",
                    help="evaluate on box-tile terrain instead of flat ground.")
parser.add_argument("--terrain-height", type=float, default=0.15,
                    help="terrain elevation in metres when --terrain (default 0.15). Use --force 0 for a "
                         "pure terrain-traversal (no-shove) survival eval.")
args = parser.parse_args()

steps = args.checkpoint * 750_000

# ---- Load the Go2 env from train.py (import by path) ----
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("go2_train", os.path.join(_here, "train.py"))
_go2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_go2)
gym.register("Go2-v0", entry_point=_go2.make_env)

ckpt_dir = args.dir if os.path.isabs(args.dir) else os.path.join(_here, args.dir)
vec_path = os.path.join(ckpt_dir, f"ppo_go2_vecnormalize_{steps}_steps.pkl")
model_path = os.path.join(ckpt_dir, f"ppo_go2_{steps}_steps.zip")
if not os.path.exists(vec_path) or not os.path.exists(model_path):
    raise SystemExit(f"[eval] no {steps:,}-step checkpoint in '{ckpt_dir}'. Check the index / --dir.")

# push=False: the env leaves xfrc_applied alone, so THIS script owns the scripted shove.
venv = make_vec_env("Go2-v0", n_envs=1,
                    env_kwargs=dict(push=False, terrain=args.terrain, terrain_height=args.terrain_height))
venv = VecNormalize.load(vec_path, venv)
venv.training = False
venv.norm_reward = False
policy = PPO.load(model_path[:-4])

mj_model = venv.get_attr("model")[0]
mj_data = venv.get_attr("data")[0]
dt = float(venv.get_attr("dt")[0])
base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")

settle_n = int(round(args.settle / dt))
dur_n = int(round(args.dur / dt))
window_n = int(round(args.window / dt))
hold_ok = int(round(HOLD_OK_S / dt))
total_n = settle_n + dur_n + window_n
shove_end = settle_n + dur_n

# 8 compass directions (unit horizontal vectors), cycled across episodes for even coverage.
COMPASS = [np.array([np.cos(a), np.sin(a), 0.0]) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]


def torso_up():
    return float(mj_data.xmat[base_id].reshape(3, 3)[2, 2])


_DOWN = np.array([0.0, 0.0, -1.0])
_RAY_GROUP = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)   # group-0 terrain (tiles / floor)
_RAY_GID = np.zeros(1, dtype=np.int32)


def local_height():
    """Torso height above the ground beneath it (a downward COM ray onto group-0 terrain), matching the
    env's h_local. On flat ground the ray hits z=0 so this equals absolute qpos[2]."""
    com = mj_data.subtree_com[base_id]
    dist = mujoco.mj_ray(mj_model, mj_data, com, _DOWN, _RAY_GROUP, 1, -1, _RAY_GID)
    ground_z = (com[2] - dist) if dist >= 0.0 else 0.0
    return float(mj_data.qpos[2] - ground_z)


print(f"[eval] {ckpt_dir}  checkpoint {args.checkpoint} -> {steps:,} steps")
print(f"[eval] {args.episodes} episodes | shove {args.force:.0f} N for {args.dur}s at t={args.settle}s, "
      f"watch {args.window}s | recover = up>={UP_OK}, h in [{H_LO},{H_HI}] for {HOLD_OK_S}s\n")

falls = 0
recovered = 0
ttu_list = []     # time-to-upright (s) for recovered episodes
dip_list = []     # lowest torso height after the shove

for ep in range(args.episodes):
    venv.seed(args.seed + ep)
    obs = venv.reset()
    mj_data.xfrc_applied[base_id, :3] = 0.0
    direction = COMPASS[ep % len(COMPASS)]
    fell = False
    min_h = np.inf
    up_run = 0
    rec = False
    t_recover = None

    for t in range(total_n):
        if settle_n <= t < shove_end:
            mj_data.xfrc_applied[base_id, :3] = args.force * direction
        elif t == shove_end:
            mj_data.xfrc_applied[base_id, :3] = 0.0

        action, _ = policy.predict(obs, deterministic=True)
        obs, _, done, info = venv.step(action)

        if done[0]:
            if not info[0].get("TimeLimit.truncated", False):
                fell = True
            break

        # local (ground-relative) height on terrain so the recovery bounds are correct over bumps;
        # on flat local_height() == absolute qpos[2], so flat eval is unchanged.
        h = local_height() if args.terrain else float(mj_data.qpos[2])
        up = torso_up()
        if t >= settle_n:
            min_h = min(min_h, h)
        if t >= shove_end:
            if up >= UP_OK and H_LO <= h <= H_HI:
                up_run += 1
                if up_run >= hold_ok and not rec:
                    rec = True
                    t_recover = ((t - hold_ok + 1) - shove_end) * dt
            else:
                up_run = 0

    mj_data.xfrc_applied[base_id, :3] = 0.0   # never carry a force into the next episode

    if fell:
        falls += 1
    elif rec:
        recovered += 1
        ttu_list.append(t_recover)
    if np.isfinite(min_h):
        dip_list.append(min_h)

    tag = "FELL" if fell else ("recovered" if rec else "stayed up, unsteady")
    print(f"  ep {ep:2d}  dir {int(np.degrees(np.arctan2(direction[1], direction[0]))):>4}deg  -> {tag}")

n = args.episodes
ground = f"terrain {args.terrain_height:.2f} m" if args.terrain else "flat ground"
lines = [
    f"Push-recovery eval @ {steps:,} steps  ({ckpt_dir})  [{ground}]",
    f"episodes {n} | shove {args.force:.0f} N for {args.dur}s at t={args.settle}s | window {args.window}s",
    "-" * 52,
    f"fall rate            : {falls / n * 100:5.1f}%   ({falls}/{n})",
    f"recovery rate        : {recovered / n * 100:5.1f}%   ({recovered}/{n})",
    f"stayed up, unsteady  : {(n - falls - recovered) / n * 100:5.1f}%   ({n - falls - recovered}/{n})",
    f"mean time-to-upright : {np.mean(ttu_list):.2f} s" if ttu_list else "mean time-to-upright : n/a (none recovered)",
    f"mean dip (min height): {np.mean(dip_list):.3f} m" if dip_list else "mean dip             : n/a",
]
report = "\n".join(lines) + "\n\n"
print("\n" + report)

out = os.path.join(ckpt_dir, f"eval_{steps}_steps.txt")
with open(out, "a") as f:
    f.write(report)
print(f"[eval] wrote {out}")
