"""
Watch a trained Unitree Go2 policy (Tier-2 PMTG) using MuJoCo's NATIVE passive viewer.

Same viewer as the legacy watch (watch_legacy.py): GPU-accelerated, renders in its own thread,
shadows off. It loads the Go2 env from THIS run's train.py, so the 16-D action / 60-D obs space
(policy-modulated per-leg phase) is handled automatically -- nothing here is action-space-specific.

Usage:  python watch.py [checkpoint] [--dir FOLDER] [--slowmo S] [--terrain] [--push]
          checkpoint : 1-4 (1=750k, 2=1.5M, 3=2.25M, 4=3M steps), 0=latest/Ctrl+C save. Default 4.
          --dir      : checkpoint folder (relative to this script, or absolute). Default checkpoints.
          --slowmo   : playback slowdown (1.0=real-time, higher=slower). Default 1.0.
        e.g.  python watch.py 4 --dir checkpoints            # the latest run, real time
              python watch.py 0 --dir checkpoints --push --slowmo 3
Run from the folder you trained in. Close the viewer window or Ctrl+C to stop.
"""
import os
import time
import argparse
import importlib.util

import numpy as np
import gymnasium as gym
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

parser = argparse.ArgumentParser(description="Watch a trained Go2 policy (native MuJoCo viewer).")
parser.add_argument("checkpoint", nargs="?", type=int, default=4,
                    help="checkpoint index 1-4 (1=750k ... 4=3M steps), 0=latest/Ctrl+C save. Default 4.")
parser.add_argument("--slowmo", type=float, default=1.0,
                    help="playback slowdown factor (1.0=real-time, higher=slower). Default 1.0.")
parser.add_argument("--terrain", action="store_true",
                    help="replay on uneven heightfield terrain instead of flat ground")
parser.add_argument("--terrain-height", type=float, default=0.06,
                    help="max terrain bump height in metres (default 0.06)")
parser.add_argument("--dir", default="checkpoints",
                    help="checkpoint folder to load from -- a name relative to this script or an "
                         "absolute path. Default: checkpoints.")
parser.add_argument("--push", action="store_true",
                    help="enable random torso shoves during playback (drawn as a yellow arrow)")
args = parser.parse_args()

checkpoint = args.checkpoint
SLOWMO = args.slowmo
steps = checkpoint * 750_000    # checkpoints are saved every 750k steps (0 -> latest/Ctrl+C save)

# ---- Load the Go2 env from train.py (import by path so this works regardless of CWD). ----
_here = os.path.dirname(os.path.abspath(__file__))
ckpt_dir = args.dir if os.path.isabs(args.dir) else os.path.join(_here, args.dir)
_spec = importlib.util.spec_from_file_location("go2_train", os.path.join(_here, "train.py"))
_go2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_go2)

gym.register("Go2-v0", entry_point=_go2.make_env)

# NOTE: no render_mode here -- we drive the native viewer ourselves, so the env does no rendering.
venv = make_vec_env("Go2-v0", n_envs=1,
                    env_kwargs=dict(terrain=args.terrain, terrain_height=args.terrain_height,
                                    push=args.push))
if args.push:
    print("[push] random torso shoves enabled (yellow arrow = active shove)")
if args.terrain:
    print(f"[terrain] uneven terrain (max bump {args.terrain_height} m) -- a flat-trained policy will stumble")
vec_path = os.path.join(ckpt_dir, f"ppo_go2_vecnormalize_{steps}_steps.pkl")
model_path = os.path.join(ckpt_dir, f"ppo_go2_{steps}_steps.zip")
if not os.path.exists(vec_path) or not os.path.exists(model_path):
    raise SystemExit(
        f"[watch] no {steps:,}-step checkpoint in '{ckpt_dir}'.\n"
        f"  looked for: {os.path.basename(model_path)} (+ vecnormalize .pkl)\n"
        f"  available here: {sorted(f for f in os.listdir(ckpt_dir) if f.endswith('.zip'))}\n"
        f"  pick a different checkpoint index, or pass --dir <folder>."
        if os.path.isdir(ckpt_dir) else
        f"[watch] checkpoint folder '{ckpt_dir}' does not exist. Pass --dir <folder>."
    )
venv = VecNormalize.load(vec_path, venv)
venv.training = False
venv.norm_reward = False

policy = PPO.load(model_path[:-4])   # PPO.load wants the path without the .zip suffix
print(f"[watch] {ckpt_dir}  checkpoint {checkpoint} -> {steps:,} steps, slowmo {SLOWMO}x")

mj_model = venv.get_attr("model")[0]
mj_data = venv.get_attr("data")[0]
dt = venv.get_attr("dt")[0]      # control timestep (0.02 s = 50 Hz)

mj_model.light_castshadow[:] = 0     # kill shadow casting -- biggest GL cost here
base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")


def draw_push(viewer, venv):
    """Draw a yellow arrow from the torso in the direction of the active shove (push training).
    Clears the viewer's user scene each frame (ngeom=0) so arrows don't accumulate."""
    scn = viewer.user_scn
    scn.ngeom = 0
    if not bool(venv.get_attr("_push_active")[0]):
        return
    force = np.asarray(venv.get_attr("_push_force")[0], dtype=np.float64)
    mag = float(np.linalg.norm(force))
    if mag < 1e-6:
        return
    scn = viewer.user_scn
    i = scn.ngeom
    if i >= len(scn.geoms):
        return
    p0 = np.asarray(mj_data.xpos[base_id], dtype=np.float64).copy()
    p1 = p0 + force / mag * 0.30
    g = scn.geoms[i]
    mujoco.mjv_initGeom(g, int(mujoco.mjtGeom.mjGEOM_ARROW),
                        np.zeros(3), np.zeros(3), np.zeros(9),
                        np.array([1.0, 0.85, 0.1, 1.0], dtype=np.float32))
    mujoco.mjv_connector(g, int(mujoco.mjtGeom.mjGEOM_ARROW), 0.015, p0, p1)
    scn.ngeom = i + 1


obs = venv.reset()
step_dt = dt * SLOWMO
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = base_id
    viewer.cam.distance = 2.5
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -12
    next_t = time.perf_counter()
    while viewer.is_running():
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, done, info = venv.step(action)
        draw_push(viewer, venv)
        viewer.sync()
        if done.any():
            if args.terrain:
                try:
                    viewer.update_hfield(0)
                except Exception:
                    pass
            viewer.sync()
            time.sleep(0.5)
            next_t = time.perf_counter()
            continue
        next_t += step_dt
        remaining = next_t - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        else:
            next_t = time.perf_counter()
