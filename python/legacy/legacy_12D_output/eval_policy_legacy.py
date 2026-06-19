"""
Quantitative robustness eval for a trained Go2 policy -- turns "watch it and judge" into numbers.

Runs the DETERMINISTIC policy over N seeded episodes. Each episode lets the robot reach a steady gait,
then applies a SCRIPTED external shove to the torso (fixed timing + magnitude; direction cycled over
the 8 compass points for coverage) and measures whether it recovers. Reports over all episodes:
  * fall rate            -- fraction that terminate (fall) after the shove
  * recovery rate        -- fraction that return to a steady upright stance and hold it
  * mean time-to-upright -- avg seconds from shove-end to sustained-upright (recovered episodes only)
  * mean dip             -- avg lowest torso height reached after the shove (how far it sagged)
The same numbers are written to eval_<steps>_steps.txt in the checkpoint folder, for the research log.

Run the SAME eval on the pre-push trot and the push-trained policy: the drop in fall rate / rise in
recovery rate is the quantitative evidence the push run worked (and the readiness signal for terrain).

Usage:
  python eval_policy.py 4 --dir checkpoints-final-gait                 # eval the 3M trot
  python eval_policy.py 0 --dir checkpoints --episodes 32 --force 90   # a push-trained run's latest save
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
parser.add_argument("--force", type=float, default=90.0, help="scripted shove force in N (default 90).")
parser.add_argument("--dur", type=float, default=0.2, help="scripted shove duration in s (default 0.2).")
parser.add_argument("--settle", type=float, default=2.0, help="settle time before the shove, s (default 2.0).")
parser.add_argument("--window", type=float, default=3.0, help="recovery window after the shove, s (default 3.0).")
parser.add_argument("--seed", type=int, default=0, help="base RNG seed (default 0).")
args = parser.parse_args()

steps = args.checkpoint * 750_000

# ---- Load the Go2 env from train_legacy.py (import by path) ----
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("go2_train", os.path.join(_here, "train_legacy.py"))
_go2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_go2)
gym.register("Go2-v0", entry_point=_go2.make_env)

ckpt_dir = args.dir if os.path.isabs(args.dir) else os.path.join(_here, args.dir)
vec_path = os.path.join(ckpt_dir, f"ppo_go2_vecnormalize_{steps}_steps.pkl")
model_path = os.path.join(ckpt_dir, f"ppo_go2_{steps}_steps.zip")
if not os.path.exists(vec_path) or not os.path.exists(model_path):
    raise SystemExit(f"[eval] no {steps:,}-step checkpoint in '{ckpt_dir}'. Check the index / --dir.")

# push=False: the env leaves xfrc_applied alone, so THIS script owns the scripted shove.
venv = make_vec_env("Go2-v0", n_envs=1, env_kwargs=dict(push=False))
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
            # done from termination = a fall; done from the 1000-step TimeLimit = survived.
            if not info[0].get("TimeLimit.truncated", False):
                fell = True
            break

        h = float(mj_data.qpos[2])
        up = torso_up()
        if t >= settle_n:
            min_h = min(min_h, h)
        if t >= shove_end:
            if up >= UP_OK and H_LO <= h <= H_HI:
                up_run += 1
                if up_run >= hold_ok and not rec:
                    rec = True
                    t_recover = ((t - hold_ok + 1) - shove_end) * dt   # s from shove-end to sustained-upright
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
lines = [
    f"Push-recovery eval @ {steps:,} steps  ({ckpt_dir})",
    f"episodes {n} | shove {args.force:.0f} N for {args.dur}s at t={args.settle}s | window {args.window}s",
    "-" * 52,
    f"fall rate            : {falls / n * 100:5.1f}%   ({falls}/{n})",
    f"recovery rate        : {recovered / n * 100:5.1f}%   ({recovered}/{n})",
    f"stayed up, unsteady  : {(n - falls - recovered) / n * 100:5.1f}%   ({n - falls - recovered}/{n})",
    f"mean time-to-upright : {np.mean(ttu_list):.2f} s" if ttu_list else "mean time-to-upright : n/a (none recovered)",
    f"mean dip (min height): {np.mean(dip_list):.3f} m" if dip_list else "mean dip             : n/a",
]
report = "\n".join(lines) + "\n"
print("\n" + report)

out = os.path.join(ckpt_dir, f"eval_{steps}_steps.txt")
with open(out, "w") as f:
    f.write(report)
print(f"[eval] wrote {out}")
