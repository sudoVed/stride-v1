"""
Unitree Go2 locomotion: SB3 PPO with PD POSITION CONTROL.

The policy outputs a normalized offset (action in [-1,1]) around the default standing pose; a PD
law tau = Kp*(q_target - q) - Kd*qd, recomputed every physics substep, converts it to torque.
Commanding the default pose yields restoring torque automatically, so the robot holds itself up
and the policy only has to learn the walking deltas. The reward terms, the 1000-step episode
limit, and checkpointing (+ per-checkpoint reward breakdown) are all configured below.

Per-run changes, training output, and analysis live in report_log.md -- not in this file.

Prereqs:
  pip install mujoco "gymnasium[mujoco]" "stable-baselines3>=2.3.0" tensorboard
Run:   python 02_train_go2.py
Watch: python watch_go2.py
"""
import os
import shutil
import argparse
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList

MENAGERIE = r"E:\STRIDE\mujoco_menagerie"
GO2_SCENE = os.path.join(MENAGERIE, "unitree_go2", "scene.xml")
GO2_SCENE_TERRAIN = os.path.join(MENAGERIE, "unitree_go2", "scene_terrain.xml")  # uneven heightfield


def _make_heightfield(nrow, ncol, rng, coarse=10, smooth_passes=6, flat_frac=0.16, feather=0.12):
    """LOW-FREQUENCY smooth heightmap in [0,1] for the terrain hfield.

    Generating from a small `coarse`x`coarse` random grid (upsampled, then box-blurred) yields wide,
    gentle, walkable bumps with BOUNDED SLOPE -- not the near-vertical faces a per-cell random field
    produces at high amplitude, which are what make a small foot clip into the heightfield and jitter.
    A FEATHERED flat patch at the centre (radial mask: 0 inside `flat_frac`, ramping to 1 over
    `feather` -- no cliff) gives the robot level ground to spawn on and walk off of smoothly.
    """
    g = rng.random((coarse, coarse))                      # low-frequency base
    ri = np.linspace(0, coarse - 1, nrow).astype(int)
    ci = np.linspace(0, coarse - 1, ncol).astype(int)
    h = g[np.ix_(ri, ci)]                                 # upsample to full hfield resolution
    for _ in range(smooth_passes):                        # blur the blocky steps into gentle hills
        h = (h + np.roll(h, 1, 0) + np.roll(h, -1, 0)
               + np.roll(h, 1, 1) + np.roll(h, -1, 1)) / 5.0
    h -= h.min()
    h /= (h.max() + 1e-9)
    yy, xx = np.meshgrid(np.linspace(-1, 1, ncol), np.linspace(-1, 1, nrow))
    r = np.sqrt(xx * xx + yy * yy)
    h *= np.clip((r - flat_frac) / feather, 0.0, 1.0)     # feathered flat spawn patch (no cliff)
    return h

FOOT_GEOMS = ["FL", "FR", "RL", "RR"]
MAX_EPISODE_STEPS = 1000

# Spawn settle / "grace" window (Run: spawn immunity). On reset the robot is spawned slightly
# ABOVE its stance (SPAWN_DROP) so it drops, and for SPAWN_HOLD_SECONDS the env COMMANDS the
# default pose (ignoring the policy) with termination + movement rewards suspended -- it falls,
# settles into a clean upright stance, THEN the policy takes over. Kills the spawn-transient blow-up.
SPAWN_HOLD_SECONDS = 0.5
SPAWN_DROP = 0.05         # m above stance to spawn on flat; robot drops & settles during the hold

# Push / disturbance training (--push, Run: robustness). Random external shoves to the torso so the
# policy learns to absorb impulses and recover balance. Applied as a world-frame force on the base
# body over a short window (a real "shove"), random horizontal direction + magnitude, spaced at
# random intervals; NEVER during the spawn-settle hold. Reward is unchanged -- recovery is already
# paid for by the orientation / height / velocity-tracking terms.
PUSH_FORCE_MIN = 30.0     # N : weakest horizontal shove  (~0.2x body weight)
PUSH_FORCE_MAX = 120.0    # N : strongest horizontal shove (~0.8x body weight)
PUSH_DURATION_S = 0.2     # s : how long each shove is applied
PUSH_INTERVAL_MIN = 2.0   # s : min gap between shoves
PUSH_INTERVAL_MAX = 4.0   # s : max gap between shoves

# Disturbance-gated "recovery mode" (push-recovery run). An off-balance signal d in [0,1] rises with
# torso tilt and roll/pitch speed. When d is high the robot is being knocked, so the tidy-gait
# penalties (pose, hip-abduction) are RELAXED and the height target is LOWERED -- licensing the crouch
# and wide foot placement that catch a shove. Deadzones keep d=0 during calm walking, so the clean
# trot is left untouched; recovery mode only engages when actually disturbed.
D_TILT_DEAD = 0.05    # tilt (1-up) below this = calm (up > 0.95); no recovery mode
D_TILT_REF = 0.30     # tilt above the deadzone at which d saturates to 1
D_RATE_DEAD = 1.5     # rad/s : normal walking roll/pitch buck below this = calm
D_RATE_REF = 4.0      # rad/s : extra roll/pitch speed above the deadzone at which d saturates to 1
HEIGHT_DROP_MAX = 0.10  # m : how much lower the height target goes when fully off-balance (lower COM)
N_ENVS = 8
TOTAL_STEPS = 3_000_000
CKPT_DIR = "./checkpoints/"   # wiped at the start of every run (see main)
# Warm-start: continue training from an existing policy instead of random init.
# Set to (model_zip, vecnormalize_pkl) or None for from-scratch. The source MUST live outside
# CKPT_DIR (which is wiped every run) -- e.g. a renamed, preserved checkpoint folder.
WARM_START = ("./checkpoints-push/ppo_go2_3000000_steps.zip",
              "./checkpoints-push/ppo_go2_vecnormalize_3000000_steps.pkl")

# PD controller gains and how far the policy may push joints off the default pose.
KP = 30.0           # position gain
KD = 0.75           # velocity (damping) gain
ACTION_SCALE = 0.7  # action in [-1,1] -> up to +/-0.7 rad offset (raised from 0.5 for taller stance)
TARGET_HEIGHT = 0.33  # desired torso height off the ground (m) -- keeps the torso up, not crouched
HEIGHT_TOL = 0.05     # gaussian tolerance / std (m): reward falls off ~5 cm either side of target

# Stepping / posture shaping -----------------------------------------------
AIR_TIME_TARGET = 0.2     # s: swing time a step must clear to earn positive air-time reward
FOOT_CLEAR_TARGET = 0.08  # m: desired swing-foot clearance above ground (bottom of foot)
FOOT_CLEAR_VREF = 0.30    # m/s: horizontal foot speed at which the clearance reward saturates
W_CLEARANCE = 0.10        # swing-foot lift incentive; small so it doesn't over-reward big leg swings
W_POSE = 0.02             # weight on default-pose L2 regularization. KEEP SMALL or it suppresses gait.

# Motion-smoothness penalties (suppress torso bucking and joint overmovement) ---
W_ACTRATE = 3e-4          # penalize fast action changes -> smoother joint motion
W_JOINTVEL = 1.5e-3       # anti-thrash on joint velocities -> discourages overmovement
W_ANGVEL = 0.05           # penalize torso roll+pitch angular velocity -> fights bucking directly
W_YAW = 0.15              # penalize torso yaw rate -> stops the body turning/twisting
W_ABDUCT = 0.5            # penalize hip-abduction deviation -> keep legs under body (in-plane gait)

# Terrain contact: a fat contact margin on the ground so feet/legs are repelled just BEFORE they
# reach the surface -> they skim over bumps instead of clipping into the heightfield and sticking.
# (Trade-off: a swing foot may visibly hover up to ~this margin above the ground. Tune as needed.)
TERRAIN_MARGIN = 0.015    # metres

# Diagonal-trot contact schedule -------------------------------------------
# Diagonal pairs: A = FL + RR, B = FR + RL. A is in stance for the first half of each gait
# cycle (B swings), then they swap. A phase clock drives the schedule and is fed to the policy.
GAIT_FREQ = 1.5           # gait cycles per second (each diagonal pair swings once per cycle)
W_GAIT = 0.15             # small diagonal-trot schedule reward, GATED by (1-d): polishes calm-gait
                          # symmetry (kills the lone-FL pre-hop) but fades during recovery. 0 = off.
PAIR_A = [0, 3]           # FOOT_GEOMS index: FL, RR
PAIR_B = [1, 2]           # FOOT_GEOMS index: FR, RL


class Go2Env(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 50}

    def __init__(self, terrain=False, terrain_height=0.06, push=False, **kwargs):
        obs_dim = 1 + 4 + 12 + 3 + 3 + 12 + 1 + 12 + 2   # = 50 (see _get_obs; +2 = gait phase sin/cos)
        observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float64)
        # Flat by default (unchanged behaviour). terrain=True loads the heightfield scene; obs/action
        # are identical, so any flat-trained policy still loads -- it just hasn't learned terrain yet.
        scene = GO2_SCENE_TERRAIN if terrain else GO2_SCENE
        super().__init__(
            scene,
            frame_skip=10,                      # 50 Hz control
            observation_space=observation_space,
            default_camera_config={"distance": 3.0},
            **kwargs,
        )
        self._terrain = bool(terrain)
        self._terrain_height = float(terrain_height)
        if terrain:
            # Override the placeholder max-elevation and fill the heightfield with a smooth random map.
            self.model.hfield_size[0, 2] = self._terrain_height
            nr, nc = int(self.model.hfield_nrow[0]), int(self.model.hfield_ncol[0])
            self.model.hfield_data[:] = _make_heightfield(nr, nc, np.random.default_rng()).flatten()
            self._spawn_lift = 0.06      # clear the contact margin + reset noise, then settle onto the flat patch
            # Fat contact margin on the ground geom -> feet/legs get a repelling force just before
            # touching, so they can't clip into the heightfield and get stuck (the pair margin is the
            # MAX of the two geoms', so setting the floor covers every robot geom that touches it).
            floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            self.model.geom_margin[floor_gid] = TERRAIN_MARGIN
            # A downward ray from the robot COM finds the ground beneath it. geomgroup mask = group 0
            # so the ray hits ONLY the terrain floor (robot geoms are groups 2/3), never a leg.
            self._ray_group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
            self._ray_geomid = np.zeros(1, dtype=np.int32)
        else:
            self._spawn_lift = SPAWN_DROP    # flat: spawn a touch high so it drops & settles in the hold
        self._base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self._foot_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FOOT_GEOMS
        ]
        # Foot collision geoms are spheres; size[0] = radius -> foot-bottom height = z - radius.
        self._foot_radius = self.model.geom_size[self._foot_geom_ids, 0].copy()
        # Nominal foot sliding friction (baseline for per-episode friction randomization on terrain).
        self._foot_friction0 = self.model.geom_friction[self._foot_geom_ids, 0].copy()
        # Torso height above the LOCAL ground beneath the robot (set every step by _local_height()).
        self._h_local = float(self.model.key_qpos[0][2])
        self._ray_origin = np.zeros(3)     # COM ray origin, exposed for the watch.py ray viz
        self._ray_dist = self._h_local     # COM-to-ground ray distance, for the watch.py ray viz
        # Default standing joint angles, taken from the model's "home" keyframe.
        self._default_joint = self.model.key_qpos[0][7:].copy()   # 12 values
        # Indices (in the 12-joint vector) of the hip ABduction joints (the sideways "*_hip_joint"
        # ones). Penalizing their deviation keeps the legs swinging in the sagittal plane instead
        # of splaying outward. Detected by name, with the standard Go2 ordering as a fallback.
        _abduct = [k for k in range(self.model.nu)
                   if "hip" in (mujoco.mj_id2name(
                       self.model, mujoco.mjtObj.mjOBJ_JOINT, self.model.actuator_trnid[k, 0]) or "").lower()]
        self._abduct_idx = np.array(_abduct if _abduct else [0, 3, 6, 9], dtype=int)
        # Torque clamp = the motors' real ctrl ranges.
        self._ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()

        # The POLICY's action space is now a normalized offset, NOT raw torque.
        self.action_space = spaces.Box(-1.0, 1.0, (self.model.nu,), np.float32)

        self._v_target = 1.0
        self._prev_action = np.zeros(self.model.nu)
        self._feet_air_time = np.zeros(4)
        self._last_contact = np.ones(4, dtype=bool)
        self._prev_foot_xy = np.zeros((4, 2))   # for finite-diff horizontal foot speed
        self._phase = 0.0                        # gait-cycle phase in [0,1) (diagonal-trot clock)
        self._hold_steps = int(round(SPAWN_HOLD_SECONDS / self.dt))  # control steps to hold default pose
        self._hold = 0                           # countdown of remaining spawn-hold steps
        # Push / disturbance state (active only when push=True; see _update_push).
        self._push = bool(push)
        self._push_force = np.zeros(3)           # current world-frame shove force (exposed for viz)
        self._push_left = 0                      # control steps remaining in the active shove
        self._push_cooldown = 0                  # control steps until the next shove
        self._push_active = False                # True while a shove is being applied (for viz)
        self._push_steps = int(round(PUSH_DURATION_S / self.dt))
        self._push_int_min = int(round(PUSH_INTERVAL_MIN / self.dt))
        self._push_int_max = int(round(PUSH_INTERVAL_MAX / self.dt))

    # ---- helpers -----------------------------------------------------------
    def _feet_contact(self):
        contacts = np.zeros(4, dtype=bool)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for fi, gid in enumerate(self._foot_geom_ids):
                if c.geom1 == gid or c.geom2 == gid:
                    contacts[fi] = True
        return contacts

    def _torso_up(self):
        return self.data.xmat[self._base_id].reshape(3, 3)[2, 2]

    def _local_height(self):
        """Torso height above the ground BENEATH the robot. On flat ground this is just absolute z.
        On terrain, cast a ray straight down from the robot COM -- filtered (geomgroup) to the ground
        geom only -- to find the local surface, so bumps/dips don't bias the height reward or the
        fall check. Also stashes the ray origin/length for the watch.py visualization."""
        com = self.data.subtree_com[self._base_id]
        self._ray_origin = com.copy()
        if not self._terrain:
            self._ray_dist = float(self.data.qpos[2])
            self._h_local = float(self.data.qpos[2])
            return self._h_local
        dist = mujoco.mj_ray(self.model, self.data, com, np.array([0.0, 0.0, -1.0]),
                             self._ray_group, 1, -1, self._ray_geomid)
        if dist < 0.0:                       # ray missed (off the heightfield edge) -> assume z=0 ground
            ground_z, self._ray_dist = 0.0, float(com[2])
        else:
            ground_z, self._ray_dist = float(com[2] - dist), float(dist)
        self._h_local = float(self.data.qpos[2] - ground_z)
        return self._h_local

    def _update_push(self):
        """Advance the random-shove schedule: count down to the next shove, fire one (random
        horizontal direction + magnitude) for PUSH_DURATION_S, then resample the gap. Sets
        self._push_force (world frame), which step() writes into data.xfrc_applied."""
        if self._push_left > 0:
            self._push_left -= 1
            if self._push_left == 0:                        # shove just ended
                self._push_force[:] = 0.0
                self._push_active = False
                self._push_cooldown = int(self.np_random.integers(
                    self._push_int_min, self._push_int_max + 1))
        elif self._push_cooldown > 0:
            self._push_cooldown -= 1
            if self._push_cooldown == 0:                    # start a new shove
                mag = self.np_random.uniform(PUSH_FORCE_MIN, PUSH_FORCE_MAX)
                ang = self.np_random.uniform(0.0, 2.0 * np.pi)
                self._push_force[:] = (mag * np.cos(ang), mag * np.sin(ang), 0.0)
                self._push_left = self._push_steps
                self._push_active = True

    def _pd_simulate(self, q_target):
        """Run frame_skip physics substeps under PD control toward q_target."""
        for _ in range(self.frame_skip):
            q = self.data.qpos[7:]
            qd = self.data.qvel[6:]
            tau = KP * (q_target - q) - KD * qd
            self.data.ctrl[:] = np.clip(tau, self._ctrl_low, self._ctrl_high)
            mujoco.mj_step(self.model, self.data)

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        # Gait phase as (sin, cos) so the policy can synchronize its legs to the schedule
        # (continuous, no discontinuity at the 1->0 wrap).
        phase_rad = 2.0 * np.pi * self._phase
        return np.concatenate([
            [self._h_local], qpos[3:7], qpos[7:],   # height = above LOCAL ground (= abs z on flat)
            qvel[0:3], qvel[3:6], qvel[6:],
            [self._v_target],
            self._prev_action,
            [np.sin(phase_rad), np.cos(phase_rad)],
        ])

    # ---- core loop ---------------------------------------------------------
    def step(self, action):
        action = np.clip(np.asarray(action), -1.0, 1.0)
        # Spawn-settle hold: for the first SPAWN_HOLD_SECONDS the env commands the default pose and
        # ignores the policy, so the robot drops and settles into a clean stance before it has to act.
        holding = self._hold > 0
        if holding:
            self._hold -= 1
            q_target = self._default_joint.copy()        # ignore the policy; hold the default stance
        else:
            q_target = self._default_joint + ACTION_SCALE * action
        # External-push disturbance (training only). Schedule/apply a world-frame shove on the torso;
        # never during the settle hold. When pushes are OFF, leave xfrc_applied untouched so an
        # external driver (the eval harness) can script its own pushes.
        if self._push:
            if holding:
                self._push_force[:] = 0.0
                self._push_active = False
            else:
                self._update_push()
            self.data.xfrc_applied[self._base_id, :3] = self._push_force
        self._pd_simulate(q_target)

        vx = self.data.qvel[0]
        up = self._torso_up()
        z = self._local_height()        # height above LOCAL ground (= absolute z on flat terrain)

        vel_track = np.exp(-((vx - self._v_target) ** 2) / 0.25)

        contact = self._feet_contact()
        self._feet_air_time += self.dt
        first_contact = contact & (~self._last_contact)
        air_time_reward = float(np.sum((self._feet_air_time - AIR_TIME_TARGET) * first_contact))
        self._feet_air_time *= ~contact
        self._last_contact = contact

        # Diagonal-trot contact schedule. Advance the phase clock, then build the expected
        # stance pattern: pair A (FL+RR) in stance for the first half of the cycle, pair B
        # (FR+RL) for the second half. Reward = fraction of feet matching the schedule, centered
        # so standing or hopping (both feet of a pair in the same state) scores ~0 and only a
        # correct alternation pays. This is what turns generic lifting into an actual gait.
        self._phase = (self._phase + GAIT_FREQ * self.dt) % 1.0
        expected_stance = np.zeros(4, dtype=bool)
        if self._phase < 0.5:
            expected_stance[PAIR_A] = True       # A down, B swings
        else:
            expected_stance[PAIR_B] = True       # B down, A swings
        gait_match = float(np.mean(np.where(expected_stance, contact, ~contact)))  # [0,1]
        gait_reward = 2.0 * gait_match - 1.0     # center to [-1, 1]; 0 at the standing/hopping baseline

        # Swing-foot clearance. Reward each airborne foot for lifting toward FOOT_CLEAR_TARGET,
        # gated by horizontal foot speed so a foot held up while standing earns nothing
        # (must be a moving step). Each swing foot contributes in [0, 1].
        foot_pos = self.data.geom_xpos[self._foot_geom_ids]                       # (4, 3)
        foot_xy_speed = np.linalg.norm((foot_pos[:, :2] - self._prev_foot_xy) / self.dt, axis=1)
        self._prev_foot_xy = foot_pos[:, :2].copy()
        foot_clear = np.clip(foot_pos[:, 2] - self._foot_radius, 0.0, None)       # bottom-of-foot height
        swing = ~contact
        clearance_reward = float(np.sum(
            swing
            * np.clip(foot_clear / FOOT_CLEAR_TARGET, 0.0, 1.0)
            * np.clip(foot_xy_speed / FOOT_CLEAR_VREF, 0.0, 1.0)
        ))

        # During the spawn-settle hold the policy isn't driving the robot, so suppress every
        # MOVEMENT reward (velocity / air-time / clearance / gait); posture terms stay active so
        # settling upright is still scored.
        if holding:
            vel_track = 0.0
            air_time_reward = 0.0
            clearance_reward = 0.0
            gait_reward = 0.0

        orient_pen = 1.0 - up
        ang_vel_rp = self.data.qvel[3] ** 2 + self.data.qvel[4] ** 2   # torso roll+pitch rates (buck)
        yaw_rate = self.data.qvel[5] ** 2                              # torso yaw rate (body turning)
        action_rate = np.sum((action - self._prev_action) ** 2)
        torque_cost = np.sum(self.data.ctrl ** 2)      # actual applied torque
        joint_vel = np.sum(self.data.qvel[6:] ** 2)
        # Pull joints back toward the default (symmetric) standing pose.
        pose_dev = np.sum((self.data.qpos[7:] - self._default_joint) ** 2)
        # Keep the hip-abduction joints near neutral (legs under body, no outward splay).
        abduct_dev = np.sum((self.data.qpos[7:][self._abduct_idx]
                             - self._default_joint[self._abduct_idx]) ** 2)
        # Off-balance signal d_off in [0,1]: how hard the robot is being disturbed, from torso tilt and
        # roll/pitch speed (both proprioceptive -> works in deployment too). Deadzones keep d_off = 0
        # during calm walking, so the clean trot is untouched; it ramps up only when actually shoved.
        tilt = 1.0 - up
        rate = np.sqrt(ang_vel_rp)
        d_off = float(np.clip(max((tilt - D_TILT_DEAD) / D_TILT_REF,
                                  (rate - D_RATE_DEAD) / D_RATE_REF), 0.0, 1.0))
        relax = 1.0 - d_off                                  # fades the tidy-gait penalties when disturbed
        # Lower the height TARGET when off-balance -> rewards dropping the COM to ride out a shove.
        target_h = TARGET_HEIGHT - HEIGHT_DROP_MAX * d_off
        # Height tracking: peaks at target_h, falls off either side (can't be gamed by hopping).
        height_track = np.exp(-((z - target_h) ** 2) / (2 * HEIGHT_TOL ** 2))

        reward = (
            1.5 * vel_track
            + 0.5 * height_track          # stand tall at ~0.33 m, not crouched
            + 0.1
            + 0.5 * air_time_reward
            + W_CLEARANCE * clearance_reward   # lift swing feet (encourages real, hind-leg stepping)
            + relax * W_GAIT * gait_reward   # diagonal-trot schedule -- GATED off during recovery
            - 0.5 * orient_pen
            - W_ANGVEL * ang_vel_rp       # fight torso buck (roll+pitch angular velocity)
            - W_YAW * yaw_rate            # stop the body turning/twisting (yaw rate)
            - W_ACTRATE * action_rate
            - 1e-5 * torque_cost          # smaller: PD torques are large; don't over-penalize
            - W_JOINTVEL * joint_vel      # anti-thrash / anti-overmovement
            - relax * W_POSE * pose_dev       # toward default stance -- RELAXED when off-balance
            - relax * W_ABDUCT * abduct_dev   # legs under body -- RELAXED so it can step out to catch a shove
        )

        self._prev_action = action.copy()
        terminated = bool(z < 0.18 or up < 0.4) and not holding   # never end during the spawn hold

        info = {
            "vx": float(vx), "v_target": float(self._v_target), "z": float(z),
            # every reward component, signed, so they sum to the step reward:
            "r_vel": float(1.5 * vel_track),
            "r_height": float(0.5 * height_track),
            "r_alive": 0.1,
            "r_airtime": float(0.5 * air_time_reward),
            "r_clearance": float(W_CLEARANCE * clearance_reward),
            "r_gait": float(relax * W_GAIT * gait_reward),
            "r_orient": float(-0.5 * orient_pen),
            "r_angvel": float(-W_ANGVEL * ang_vel_rp),
            "r_yaw": float(-W_YAW * yaw_rate),
            "r_action_rate": float(-W_ACTRATE * action_rate),
            "r_torque": float(-1e-5 * torque_cost),
            "r_jointvel": float(-W_JOINTVEL * joint_vel),
            "r_pose": float(-relax * W_POSE * pose_dev),
            "r_abduct": float(-relax * W_ABDUCT * abduct_dev),
        }
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, False, info

    def reset_model(self):
        if self._terrain:
            # Domain randomization each episode: a fresh heightfield + new foot friction. Exposes the
            # policy to varied ground so it learns to recover instead of memorizing one surface.
            nr, nc = int(self.model.hfield_nrow[0]), int(self.model.hfield_ncol[0])
            self.model.hfield_data[:] = _make_heightfield(nr, nc, self.np_random).flatten()
            self.model.geom_friction[self._foot_geom_ids, 0] = (
                self._foot_friction0 * self.np_random.uniform(0.6, 1.4, size=len(self._foot_geom_ids)))

        qpos = self.model.key_qpos[0].copy() if self.model.nkey > 0 else self.init_qpos.copy()
        qpos += self.np_random.uniform(-0.02, 0.02, self.model.nq)
        qpos[2] += self._spawn_lift      # on terrain, start above the flat spawn patch (else 0)
        qvel = self.init_qvel + self.np_random.uniform(-0.02, 0.02, self.model.nv)
        self.set_state(qpos, qvel)
        self.data.ctrl[:] = 0.0

        self._v_target = self.np_random.uniform(0.5, 1.0)
        self._prev_action = np.zeros(self.model.nu)
        self._feet_air_time = np.zeros(4)
        self._last_contact = np.ones(4, dtype=bool)
        self._prev_foot_xy = self.data.geom_xpos[self._foot_geom_ids, :2].copy()
        self._phase = self.np_random.uniform(0.0, 1.0)   # random gait phase per episode
        self._hold = self._hold_steps    # arm the spawn-settle hold (drop + hold default pose ~1 s)
        # Reset the push schedule: clear any active force, sample the gap to the first shove.
        self._push_force[:] = 0.0
        self._push_left = 0
        self._push_active = False
        self._push_cooldown = int(self.np_random.integers(self._push_int_min, self._push_int_max + 1))
        if self._push:
            self.data.xfrc_applied[self._base_id, :3] = 0.0
        self._local_height()             # set h_local for the reset observation
        return self._get_obs()


def make_env(**kw):
    return Go2Env(**kw)


# Reward components in display order (must match the keys put in info dict in step()).
REWARD_KEYS = ["r_vel", "r_height", "r_alive", "r_airtime", "r_clearance", "r_gait",
               "r_orient", "r_angvel", "r_yaw", "r_action_rate", "r_torque", "r_jointvel",
               "r_pose", "r_abduct"]


def format_reward_breakdown(total_steps, window_steps, sums, count, vx_sum, z_sum):
    """Human-readable per-term reward breakdown saved alongside each checkpoint."""
    means = {k: sums[k] / count for k in REWARD_KEYS}
    total = sum(means.values())
    out = []
    out.append(f"Reward breakdown @ {total_steps:,} timesteps")
    out.append(f"(averaged over the last ~{window_steps:,} env-steps; {count:,} samples)")
    out.append("")
    out.append(f"{'component':<16}{'mean/step':>12}{'share':>10}")
    out.append("-" * 38)
    for k in REWARD_KEYS:
        share = (means[k] / total * 100.0) if total != 0 else 0.0
        out.append(f"{k:<16}{means[k]:>12.4f}{share:>9.1f}%")
    out.append("-" * 38)
    out.append(f"{'TOTAL':<16}{total:>12.4f}")
    out.append("")
    out.append(f"mean forward velocity : {vx_sum / count:6.3f} m/s")
    out.append(f"mean torso height     : {z_sum / count:6.3f} m")
    return "\n".join(out) + "\n"


class RewardBreakdownCallback(BaseCallback):
    """Accumulates per-term rewards and writes a .txt breakdown at each checkpoint interval."""
    def __init__(self, save_freq, save_path, name_prefix="ppo_go2"):
        super().__init__()
        self.save_freq = save_freq          # in n_calls (per-env), same units as CheckpointCallback
        self.save_path = save_path
        self.name_prefix = name_prefix
        self._reset()

    def _reset(self):
        self.sums = {k: 0.0 for k in REWARD_KEYS}
        self.vx_sum = 0.0
        self.z_sum = 0.0
        self.count = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "r_vel" not in info:         # skip any info without our reward keys
                continue
            for k in REWARD_KEYS:
                self.sums[k] += info.get(k, 0.0)
            self.vx_sum += info.get("vx", 0.0)
            self.z_sum += info.get("z", 0.0)
            self.count += 1
        if self.n_calls % self.save_freq == 0 and self.count > 0:
            os.makedirs(self.save_path, exist_ok=True)
            text = format_reward_breakdown(
                self.num_timesteps, self.save_freq * len(self.locals.get("infos", [1])),
                self.sums, self.count, self.vx_sum, self.z_sum,
            )
            fn = os.path.join(self.save_path,
                              f"{self.name_prefix}_reward_breakdown_{self.num_timesteps}_steps.txt")
            with open(fn, "w") as f:
                f.write(text)
            if self.verbose:
                print(f"[reward-breakdown] wrote {fn}")
            self._reset()
        return True


def main():
    parser = argparse.ArgumentParser(description="Train the Go2 walking policy.")
    parser.add_argument("--terrain", action="store_true",
                        help="train on uneven heightfield terrain instead of flat ground")
    parser.add_argument("--terrain-height", type=float, default=0.06,
                        help="max terrain bump height in metres (default 0.06)")
    parser.add_argument("--push", action="store_true",
                        help="enable push/disturbance training (random torso shoves) + warm-start "
                             "exploration boost, so the policy learns to recover from impulses")
    args = parser.parse_args()

    # Wipe old checkpoints FIRST so a rerun can never leave stale model/.pkl/reward-breakdown.txt
    # files from a previous run lying around to be misread later.
    if os.path.isdir(CKPT_DIR):
        shutil.rmtree(CKPT_DIR)
        print(f"[setup] wiped old checkpoints in {CKPT_DIR}")
    os.makedirs(CKPT_DIR, exist_ok=True)

    gym.register("Go2-v0", entry_point=make_env, max_episode_steps=MAX_EPISODE_STEPS)
    venv = make_vec_env("Go2-v0", n_envs=N_ENVS,
                        env_kwargs=dict(terrain=args.terrain, terrain_height=args.terrain_height,
                                        push=args.push))
    if args.terrain:
        print(f"[terrain] training on uneven terrain (max bump {args.terrain_height} m)")
    if args.push:
        print(f"[push] disturbance training on: shoves {PUSH_FORCE_MIN:.0f}-{PUSH_FORCE_MAX:.0f} N, "
              f"{PUSH_DURATION_S}s, every {PUSH_INTERVAL_MIN}-{PUSH_INTERVAL_MAX}s")

    if WARM_START:
        # Continue from a saved policy: restore its VecNormalize stats (so obs scaling matches what
        # the policy expects) and load the network weights. Hyperparameters come from the saved model.
        model_path, vec_path = WARM_START
        venv = VecNormalize.load(vec_path, venv)
        venv.training = True            # keep adapting the normalization stats
        venv.norm_reward = True
        model = PPO.load(model_path, env=venv, device="auto", tensorboard_log="runs")
        print(f"[warm-start] loaded {model_path}")
        if args.terrain or args.push:
            # The flat policy is over-converged (tiny action std), which leaves little exploration to
            # adapt with. Re-open it: raise the entropy bonus (ent_coef) so PPO is rewarded for keeping
            # some randomness, and re-inflate the action std so it actually tries variations (e.g.
            # recovery steps) instead of clinging to the flat gait. Needed for both terrain and pushes.
            model.ent_coef = 0.01
            model.policy.log_std.data.fill_(float(np.log(0.5)))
            print("[explore] exploration boost: ent_coef=0.01, action std reset to 0.5")
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = PPO(
            "MlpPolicy", venv,
            n_steps=2048, batch_size=4096, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, learning_rate=3e-4,
            clip_range=0.2, ent_coef=0.0,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=1, tensorboard_log="runs", device="auto",
        )

    save_freq = max(750_000 // N_ENVS, 1)
    ckpt = CheckpointCallback(
        save_freq=save_freq,
        save_path=CKPT_DIR,
        name_prefix="ppo_go2",
        save_vecnormalize=True,
    )
    # Writes ppo_go2_reward_breakdown_<steps>_steps.txt next to each checkpoint.
    breakdown = RewardBreakdownCallback(
        save_freq=save_freq, save_path=CKPT_DIR, name_prefix="ppo_go2",
    )
    callbacks = CallbackList([ckpt, breakdown])

    try:
        # reset_num_timesteps=True so a warm-started run still numbers its checkpoints from 750k
        # (keeps the 1-4 watch mapping), rather than continuing the loaded model's step count.
        model.learn(total_timesteps=TOTAL_STEPS, callback=callbacks, progress_bar=True,
                    reset_num_timesteps=True)
    except KeyboardInterrupt:
        print("\n[interrupted] saving current model before exit...")
    finally:
        # Save the latest model into the checkpoints folder using the SAME naming scheme as the
        # periodic checkpoints, but with step number 0. That way watch.py / watch_go2.py can load
        # this manual/interrupted save with checkpoint argument 0 (steps = 0 * 750000 = 0):
        #   checkpoints/ppo_go2_0_steps.zip  +  checkpoints/ppo_go2_vecnormalize_0_steps.pkl
        model.save(os.path.join(CKPT_DIR, "ppo_go2_0_steps"))
        venv.save(os.path.join(CKPT_DIR, "ppo_go2_vecnormalize_0_steps.pkl"))
        print(f"Saved latest model -> {CKPT_DIR}ppo_go2_0_steps.zip (+ vecnormalize). Watch it with: 0")
        # Also write a reward-breakdown .txt for this latest/interrupted save (step 0), like the
        # periodic checkpoints, from whatever the callback has accumulated since its last write.
        if breakdown.count > 0:
            fn = os.path.join(CKPT_DIR, "ppo_go2_reward_breakdown_0_steps.txt")
            with open(fn, "w") as f:
                f.write(format_reward_breakdown(
                    model.num_timesteps, breakdown.count, breakdown.sums,
                    breakdown.count, breakdown.vx_sum, breakdown.z_sum))
            print(f"Saved reward breakdown -> {fn}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Tuning knobs for future runs (log each change as a new Run):
#   KP / KD       -> stiffer legs (higher KP) hold pose better but can get twitchy
#   ACTION_SCALE  -> how big a stride the policy can command (0.3 cautious, 0.7 bold)
#   reward weights-> see notes in report_log.md
# ---------------------------------------------------------------------------