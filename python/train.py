"""
Unitree Go2 locomotion -- Tier-3 PMTG: Foot Trajectory Generator + policy residuals.  [Run 18+]

Carried over from the flat-robustness line (legacy: train_legacy.py / Runs 1-15): SB3 PPO with PD
POSITION CONTROL. A PD law tau = Kp*(q_target - q) - Kd*qd, recomputed every physics substep,
converts a joint position target into torque, so the robot holds itself up "for free."

WHY TIER 3 (the ANYmal / PMTG architecture) -- and why Tier 2 failed
--------------------------------------------------------------------
Tier 2 (Runs 16-17) gave the policy four INDEPENDENT per-leg phase clocks and tried to make BOTH the
swing arc and the inter-leg coordination EMERGE from rewards (a clearance reward + a per-leg
phase-consistency reward). It failed in exactly the way the ANYmal papers predict: Run 16 wouldn't
lift the feet (a low "strut"); cranking the clearance/phase weights in Run 17 lifted the feet but the
four free clocks drifted off the trot relationship, giving an uncoordinated, stumbling high-step.

Lee et al. 2020 ("Learning quadrupedal locomotion over challenging terrain", ANYmal-C) and the PMTG
formulation it builds on (Iscen et al. 2018) DON'T make those things emerge -- they make them
STRUCTURAL:
  * A SHARED base clock, not free per-leg clocks. One common base frequency `BASE_FREQ`; each leg's
    phase is `phi_i = (phi_i0 + (BASE_FREQ + df_i)*t) mod 1`, and the policy only outputs a bounded
    per-leg frequency OFFSET `df_i`. The legs cannot free-run apart.
  * The TROT COORDINATION lives in fixed initial phase offsets `phi_i0` (FL+RR vs FR+RL), a structural
    prior -- not something a reward must discover or hold together.
  * A FOOT TRAJECTORY GENERATOR (FTG) maps each leg's phase to a NOMINAL swing motion, so the foot
    lifts and steps BY CONSTRUCTION; the policy outputs only RESIDUALS on top. (Lee 2020: action =
    leg frequencies + foot-position residuals; "the FTG drives the vertical stepping motion".)
We keep their structure but adapt the residual to OUR joint-space PD setup: the FTG emits a nominal
per-leg JOINT offset (a knee-bend + thigh-lift swing), and the policy's 12 joint dims are residuals
on top -- no inverse kinematics needed.

  q_target = default_pose + FTG(phase) + RESIDUAL_SCALE * action[:12]
  phase_i   = (global_clock + TROT_OFFSET_i + delta_i) % 1        # global_clock += BASE_FREQ*dt
  delta_i  += FREQ_RANGE * action[12+i] * dt                      # CLAMPED to +/-DELTA_MAX, recentered when calm

Action stays 16-D (12 joint residuals + 4 per-leg frequency offsets) and obs stays 60-D, so
watch.py / eval_policy.py are unchanged. Coordination + swing are now structural; the d-gated
recovery, spawn hold, push DR and all other reward terms are unchanged. Trains from scratch
(WARM_START = None). See report_logs_3.md for the per-run analysis.

Prereqs:  pip install mujoco "gymnasium[mujoco]" "stable-baselines3>=2.3.0" tensorboard
Run:   python train.py [--push] [--terrain [--terrain-height H]]
Watch: python watch.py ;  Eval: python eval_policy.py
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


# --- Terrain strip (must match make_box_terrain.py) ------------------------------------------------
# Forward strip: tiles span x in [TERRAIN_XB, TERRAIN_XF] (mostly forward, the +x walking direction) and
# y in [-TERRAIN_YH, TERRAIN_YH]. Half the tiles aren't wasted behind the robot, and the long forward
# reach stops it walking off the edge mid-episode.
TERRAIN_XB, TERRAIN_XF, TERRAIN_YH, TERRAIN_PITCH = -1.5, 18.0, 4.0, 0.5
# Fractal value-noise octaves (period_m, amplitude): a big rolling hill + medium + a little texture.
# Tuned (offline) so adjacent-tile steps stay ~20 mm (well under the foot's bridge size) while giving
# ~10 cm of varied elevation -- the old single-scale blur was near-Gaussian and barely changed.
#TERRAIN_OCTAVES = [(9.0, 1.0), (4.5, 0.30), (2.2, 0.12)]
TERRAIN_OCTAVES = [(4.0, 1.0), (2.0, 0.6), (1.0, 0.4)]   # high-freq weighted -> rougher, more consistent variation
TERRAIN_FLAT_R, TERRAIN_FLAT_F = 0.5, 2.5      # flat spawn patch radius + feather ramp (m), centred at origin


def _value_noise(x, y, period, rng):
    """Bilinear value noise at the given spatial period, sampled at world (x, y). LINEAR interpolation
    (no smoothstep) so the terrain has constant-slope ramps between grid nodes instead of flattening to
    zero slope at every node -> higher, more consistent mean slope."""
    xspan, yspan = TERRAIN_XF - TERRAIN_XB, 2.0 * TERRAIN_YH
    gx = max(2, int(np.ceil(xspan / period)) + 2)
    gy = max(2, int(np.ceil(yspan / period)) + 2)
    g = rng.random((gy, gx))
    fx = (x - TERRAIN_XB) / period
    fy = (y + TERRAIN_YH) / period
    x0 = np.clip(np.floor(fx).astype(int), 0, gx - 2)
    y0 = np.clip(np.floor(fy).astype(int), 0, gy - 2)
    tx = np.clip(fx - x0, 0.0, 1.0)
    ty = np.clip(fy - y0, 0.0, 1.0)
    return (g[y0, x0] * (1 - tx) * (1 - ty) + g[y0, x0 + 1] * tx * (1 - ty)
            + g[y0 + 1, x0] * (1 - tx) * ty + g[y0 + 1, x0 + 1] * tx * ty)


def _terrain_heights(xy, rng, terrain_height):
    """Per-tile heights: multi-octave (fractal) value noise -> varied rolling terrain, with a flat
    spawn patch at the origin. xy = (N,2) tile centres -> (N,) heights in [0, terrain_height]."""
    x, y = xy[:, 0], xy[:, 1]
    h = np.zeros(len(xy)); tot = 0.0
    for period, amp in TERRAIN_OCTAVES:
        h += amp * _value_noise(x, y, period, rng); tot += amp
    h /= tot
    h -= h.min(); h /= (h.max() + 1e-9); h *= terrain_height
    r = np.hypot(x, y)
    m = np.clip((r - TERRAIN_FLAT_R) / TERRAIN_FLAT_F, 0.0, 1.0)
    return h * (m * m * (3.0 - 2.0 * m))       # smoothstep flat-patch ramp (no sharp wall at the patch edge)

# Foot/leg order is FL, FR, RL, RR throughout (matches the model's actuator + geom ordering:
# each leg is [hip(abduction), thigh, calf], legs in this order). Index i indexes both legs and feet.
FOOT_GEOMS = ["FL", "FR", "RL", "RR"]
N_LEGS = 4
MAX_EPISODE_STEPS = 1000

# Spawn settle / "grace" window (Run 13). Spawn ~5 cm high, command the default pose for
# SPAWN_HOLD_SECONDS with termination + movement rewards suspended, so the robot settles before acting.
SPAWN_HOLD_SECONDS = 0.5
SPAWN_DROP = 0.05

# Push / disturbance training (--push, Runs 13-14): random world-frame torso shoves.
PUSH_FORCE_MIN = 30.0
PUSH_FORCE_MAX = 120.0
PUSH_DURATION_S = 0.2
PUSH_INTERVAL_MIN = 2.0
PUSH_INTERVAL_MAX = 4.0

# Disturbance-gated "recovery mode" (Run 14): off-balance signal d in [0,1] from torso tilt +
# roll/pitch speed (deadzoned). High d relaxes the tidy-gait penalties and lowers the height target.
D_TILT_DEAD = 0.05
D_TILT_REF = 0.30
D_RATE_DEAD = 1.5
D_RATE_REF = 4.0
HEIGHT_DROP_MAX = 0.10

N_ENVS = 8
TOTAL_STEPS = 3_000_000
CKPT_DIR = "./checkpoints/"   # wiped at the start of every run (see main)

# --- Warm-start (continue from a saved policy instead of random init) ------------------------------
# HOW TO TOGGLE:
#   ON  -> WARM_START = (model_zip, vecnormalize_pkl) of a PRESERVED policy. The folder MUST live
#          OUTSIDE CKPT_DIR (which is wiped every run). Same action/obs space only (Tier-3 -> Tier-3).
#   OFF -> WARM_START = None  (train from scratch).
# `legacy/legacy_16D_output/run25_ftg/` is the clean flat Tier-3 trot (Run 25, was `checkpoints_ftg/`);
# `legacy/legacy_16D_output/run27_push_ftg/` is the d-gated push policy (Run 27) -> warm-start --terrain
# (Run 28) from it. Paths are relative to python/ (run train.py from there).
# On a --push or --terrain warm-start the code also raises ent_coef to 0.01 and resets the action std
# to 0.5 (exploration boost), so the over-converged flat policy has room to learn the new skill.
WARM_START = ("./legacy/legacy_16D_output/run27_push_ftg/ppo_go2_3000000_steps.zip",
              "./legacy/legacy_16D_output/run27_push_ftg/ppo_go2_vecnormalize_3000000_steps.pkl")
# WARM_START = None   # <-- uncomment this (and comment the two lines above) to train from scratch

# PD controller gains.
KP = 30.0
KD = 0.75
TARGET_HEIGHT = 0.33
HEIGHT_TOL = 0.05
W_VEL = 1.0              # weight on velocity-tracking (Run 24: 1.5->1.0; it was saturating the gradient)
W_HEIGHT = 0.5           # weight on height-tracking (Run 24: 0.8->0.5; keep ~2:1 vel:height so it still walks)

# Stepping / posture shaping. Clearance + phase are now mostly handled STRUCTURALLY by the FTG, so
# these reward weights are AUXILIARY (encourage a little extra lift / keep contacts honest), not the
# primary drivers they had to be in Tier 2.
# r_airtime (restored Run 25): paid at each touchdown, PEAKED at a target flight time (not the legacy
# unbounded linear). Same target for all four legs -> pulls every leg to the SAME swing duration, which
# is what enforces symmetry (a dragging rear has too-short flight; a hopping leg too-long -> both lose).
AIR_TIME_TARGET = 0.25     # s : target swing flight time (FTG swing half at 1.5 Hz ~ 0.33 s)
AIR_TOL = 0.08             # s : gaussian tolerance around the target
W_AIRTIME = 0.5            # weight on the airtime target reward
FOOT_CLEAR_TARGET = 0.10   # m: desired swing-foot clearance for the (auxiliary) clearance reward
FOOT_CLEAR_VREF = 0.30
W_CLEARANCE = 0.10         # auxiliary now (the FTG lifts the feet); Run 23: reset 0.20 -> 0.10
W_POSE = 0.02

# FTG-tracking shortfall penalty (Run 23). The FTG is an additive joint offset the policy can CANCEL
# with an opposing residual, and r_clearance is an optional bonus it can decline -- so the policy kept
# the (propulsive) rear legs flat. This term makes following the FTG's swing a REQUIREMENT, not a bonus:
# each swing foot is penalized for falling SHORT of the FTG's nominal foot clearance. Shortfall-only
# (lifting higher than nominal is free) + d-gated, so terrain deviation stays cheap (see report_logs_3).
FTG_FOOT_LIFT = 0.09       # m: the FTG's nominal peak foot clearance (FK-measured ~0.094)
W_TRACK = 3.0              # weight on the per-leg swing-clearance shortfall penalty

# Motion-smoothness penalties.
W_ACTRATE = 3e-4
W_JOINTVEL = 1.5e-3
W_ANGVEL = 0.05
W_YAW = 0.15
W_ABDUCT = 0.5

# --- Tier-3 PMTG: shared base clock + fixed trot offsets + FTG, with a CLAMPED per-leg offset -------
# ONE shared global clock advances at BASE_FREQ. Each leg's phase = (global_clock + trot_offset_i +
# delta_i) mod 1, where delta_i is a per-leg phase offset the policy ACCUMULATES via the frequency
# action (delta_i += FREQ_RANGE * a_i * dt) -- so a_i still controls each leg's phase RATE (can stall/
# advance a leg, the ANYmal frequency mechanism). The twist vs Run 18: delta_i is CLAMPED to
# [-DELTA_MAX, DELTA_MAX] so the per-leg phase can never drift more than DELTA_MAX from its trot slot
# (coordination is now a hard bound, not a soft hope -- fixes Run 18's drift into a lateral gait), and
# when calm it is bled back toward 0 (RECENTER, d-gated) so flat walking returns to a clean diagonal
# while retiming stays fully available under disturbance / on terrain.
BASE_FREQ = 1.5            # Hz : shared global clock rate (the trot cadence for all four legs)
FREQ_RANGE = 1.5           # Hz : how fast the freq action drives the per-leg offset (rate of delta change)
DELTA_MAX = 0.10           # cycles : hard clamp on |per-leg phase offset from trot| WHEN CALM (Run 20: 0.15->0.10)
# Run 27 (train_test): d-gated phase clamp. The CALM clamp (DELTA_MAX) keeps a clean trot, but a hard
# bound that never releases also forbids the big leg retiming a real push recovery needs (a leg can't
# stall/rush its phase to plant a foot under the COM). So the effective clamp WIDENS with the off-balance
# signal d: DELTA_MAX_eff = DELTA_MAX + (DELTA_MAX_PUSH - DELTA_MAX) * d. d=0 -> identical to before
# (clean flat trot untouched); d=1 -> up to DELTA_MAX_PUSH, restoring the foot-trapping / leg-stall reflex
# exactly when off-balance. (Also the lever terrain wants -- handoff sec 6.4.)
DELTA_MAX_PUSH = 0.30      # cycles : the clamp bound when FULLY off-balance (d=1); >= DELTA_MAX
RECENTER = 0.02            # per-step pull of delta -> 0 when calm (d-gated); ~1.5 s decay; 0 = no recenter
# Fixed trot phase offsets -> diagonal pairs FL+RR (0.0) and FR+RL (0.5). THIS is the inter-leg
# coordination, structural (not a reward). Order: FL, FR, RL, RR.
TROT_OFFSET = np.array([0.0, 0.5, 0.5, 0.0])

# Foot Trajectory Generator. During the SWING half of a leg's phase (phi in [0.5, 1)) the FTG adds a
# nominal joint offset that lifts the foot (knee bend + thigh lift), peaking mid-swing; ZERO during the
# STANCE half (phi in [0, 0.5)) so the foot rests at the default stance. Amplitudes are in radians and
# were tuned (see report_logs_3.md) so the nominal foot lift is ~0.09 m with the policy residual at 0.
FTG_THIGH_AMP = 0.29       # rad : peak thigh (hip-pitch) lift during swing  (tuned -> ~0.10 m foot lift)
FTG_CALF_AMP = -0.55       # rad : peak calf (knee) flexion during swing (more-negative = knee bends up)
# Policy joint residual scale -- how far the policy may push joints off the FTG+default nominal.
# Run 21: reverted 0.7 -> 0.5. Run 20's 0.7 (with r_pose dropped) let the policy override the FTG with
# jittery residuals -> stutter + lost FTG-tracking. The FTG provides the gross motion; residual refines.
# Run 27 (train_test): flat 0.5 -> 0.6, a modest bump in action range so the policy has more room to brace
# / reposition under disturbance. Kept well under Run 20's 0.7 (which caused stutter), and r_pose stays on.
RESIDUAL_SCALE = 0.5

# Structural stance-lift (Run 21). The default keyframe pose only STANDS at ~0.288 m, so the body sat
# crouched and no amount of height-reward / residual authority lifted it (Run 20). Instead we raise the
# NOMINAL stance: add a constant per-leg offset (straighten the legs -> taller) to the default joints,
# so the robot stands at ~0.32 m BY CONSTRUCTION and the FTG swings (and r_pose regularizes) from that
# taller base. FK-tuned: thigh -0.10, calf +0.20 rad -> nominal standing height 0.288 -> 0.320 m.
STANCE_LIFT_THIGH = -0.10  # rad : per-leg thigh offset added to the default pose (straighten -> taller)
STANCE_LIFT_CALF = 0.20    # rad : per-leg calf offset (less-negative = knee straighter -> taller)

# Phase-consistency reward (auxiliary): reward each leg's contact matching the swing/stance its phase
# implies. With coordination structural, this just keeps contacts honest to the FTG schedule. d-gated.
W_PHASE = 0.15
# Tiny regularizer on the frequency action (discourages needless clock churn when calm); the hard
# DELTA_MAX clamp + RECENTER are what actually guarantee coordination. d-gated. 0 = off.
W_FREQ = 0.02


def _ftg_offset(phase):
    """Map the 4 leg phases -> a 12-D nominal joint offset (the swing arc). Vectorized.
    Lift profile per leg: 0 during stance (phi<0.5), a sin bump during swing (phi in [0.5,1))."""
    s = np.clip((phase - 0.5) / 0.5, 0.0, 1.0)        # swing progress in [0,1] (0 during stance)
    lift = np.where(phase < 0.5, 0.0, np.sin(np.pi * s))   # (4,) : 0 in stance, 0->1->0 over swing
    off = np.zeros(3 * N_LEGS)
    for i in range(N_LEGS):
        off[3 * i + 1] = FTG_THIGH_AMP * lift[i]      # thigh
        off[3 * i + 2] = FTG_CALF_AMP * lift[i]       # calf
    return off


class Go2Env(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 50}

    def __init__(self, terrain=False, terrain_height=0.15, push=False, **kwargs):
        # obs: height(1) + quat(4) + joints(12) + linvel(3) + angvel(3) + jointvel(12) + v_target(1)
        #      + prev_action(16) + per-leg phase sin/cos(8) = 60
        obs_dim = 1 + 4 + 12 + 3 + 3 + 12 + 1 + 16 + 8   # = 60
        observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float64)
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
            # Box-tile terrain as MOCAP bodies (scene_terrain.xml; make_box_terrain.py). Mocap so each
            # tile's height can be set at runtime via data.mocap_pos with LIVE collision -- moving plain
            # static geoms left the raised tiles' collision at their compile height, so feet sank into
            # them. Collect each tile body's mocap index (row-major, bodies named tile_0..).
            mids = []
            k = 0
            while True:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tile_%d" % k)
                if bid < 0:
                    break
                mids.append(int(self.model.body_mocapid[bid]))
                k += 1
            self._tile_mocap = np.array(mids, dtype=int)
            g0 = int(self.model.body_geomadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tile_0")])
            self._tile_half_z = float(self.model.geom_size[g0, 2])
            self._spawn_lift = 0.06
            self._ray_group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)  # ray hits group-0 tiles
            self._ray_geomid = np.zeros(1, dtype=np.int32)
        else:
            self._spawn_lift = SPAWN_DROP
        self._base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self._foot_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FOOT_GEOMS
        ]
        self._foot_radius = self.model.geom_size[self._foot_geom_ids, 0].copy()
        self._foot_friction0 = self.model.geom_friction[self._foot_geom_ids, 0].copy()
        self._h_local = float(self.model.key_qpos[0][2])
        self._default_joint = self.model.key_qpos[0][7:].copy()   # 12 default standing angles
        # Structural stance-lift: straighten each leg a bit so the NOMINAL stance stands ~0.32 m (was
        # ~0.288 m). The FTG swing and r_pose both reference this taller nominal (Run 21).
        for _i in range(N_LEGS):
            self._default_joint[3 * _i + 1] += STANCE_LIFT_THIGH   # thigh
            self._default_joint[3 * _i + 2] += STANCE_LIFT_CALF    # calf
        _abduct = [k for k in range(self.model.nu)
                   if "hip" in (mujoco.mj_id2name(
                       self.model, mujoco.mjtObj.mjOBJ_JOINT, self.model.actuator_trnid[k, 0]) or "").lower()]
        self._abduct_idx = np.array(_abduct if _abduct else [0, 3, 6, 9], dtype=int)
        self._ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        self._n_joints = self.model.nu   # 12

        # Action: 12 joint RESIDUALS + 4 per-leg frequency offsets = 16.
        self.action_space = spaces.Box(-1.0, 1.0, (self._n_joints + N_LEGS,), np.float32)

        self._v_target = 1.0
        self._prev_action = np.zeros(self._n_joints + N_LEGS)   # 16-D
        self._feet_air_time = np.zeros(4)
        self._last_contact = np.ones(4, dtype=bool)
        self._prev_foot_xy = np.zeros((4, 2))
        self._phase_global = 0.0                      # shared global clock in [0,1)
        self._delta = np.zeros(N_LEGS)                # per-leg phase offset from trot, clamped to +/-DELTA_MAX
        self._phase = TROT_OFFSET.copy()              # effective per-leg phase = global + trot_offset + delta
        self._relax = 1.0                             # (1-d) from the previous step, used to gate RECENTER
        self._hold_steps = int(round(SPAWN_HOLD_SECONDS / self.dt))
        self._hold = 0
        # Push / disturbance state.
        self._push = bool(push)
        self._push_force = np.zeros(3)
        self._push_left = 0
        self._push_cooldown = 0
        self._push_active = False
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
        com = self.data.subtree_com[self._base_id]
        if not self._terrain:
            self._h_local = float(self.data.qpos[2])
            return self._h_local
        dist = mujoco.mj_ray(self.model, self.data, com, np.array([0.0, 0.0, -1.0]),
                             self._ray_group, 1, -1, self._ray_geomid)
        ground_z = 0.0 if dist < 0.0 else float(com[2] - dist)
        self._h_local = float(self.data.qpos[2] - ground_z)
        return self._h_local

    def _update_push(self):
        if self._push_left > 0:
            self._push_left -= 1
            if self._push_left == 0:
                self._push_force[:] = 0.0
                self._push_active = False
                self._push_cooldown = int(self.np_random.integers(
                    self._push_int_min, self._push_int_max + 1))
        elif self._push_cooldown > 0:
            self._push_cooldown -= 1
            if self._push_cooldown == 0:
                mag = self.np_random.uniform(PUSH_FORCE_MIN, PUSH_FORCE_MAX)
                ang = self.np_random.uniform(0.0, 2.0 * np.pi)
                self._push_force[:] = (mag * np.cos(ang), mag * np.sin(ang), 0.0)
                self._push_left = self._push_steps
                self._push_active = True

    def _pd_simulate(self, q_target):
        """Run frame_skip physics substeps under PD control toward q_target (12 joints)."""
        for _ in range(self.frame_skip):
            q = self.data.qpos[7:]
            qd = self.data.qvel[6:]
            tau = KP * (q_target - q) - KD * qd
            self.data.ctrl[:] = np.clip(tau, self._ctrl_low, self._ctrl_high)
            mujoco.mj_step(self.model, self.data)

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        phase_rad = 2.0 * np.pi * self._phase                     # (4,)
        phase_feat = np.concatenate([np.sin(phase_rad), np.cos(phase_rad)])  # (8,) = [sin x4, cos x4]
        return np.concatenate([
            [self._h_local], qpos[3:7], qpos[7:],
            qvel[0:3], qvel[3:6], qvel[6:],
            [self._v_target],
            self._prev_action,        # 16-D
            phase_feat,               # 8-D
        ])

    # ---- core loop ---------------------------------------------------------
    def step(self, action):
        action = np.clip(np.asarray(action), -1.0, 1.0)
        joint_res = action[:self._n_joints]        # 12 joint residuals (on top of default + FTG)
        freq_act = action[self._n_joints:]         # 4 per-leg frequency offsets

        # Advance the SHARED global clock, then build each leg's effective phase = global + trot_offset
        # + delta_i. delta_i accumulates the policy's frequency action (so a_i sets each leg's phase
        # RATE -- the ANYmal mechanism), but is CLAMPED to +/-DELTA_MAX (coordination can't drift away)
        # and, when calm, bled toward 0 (RECENTER * relax) so flat walking returns to a clean trot.
        # During the spawn hold the policy isn't driving: global clock only, delta forced to 0.
        self._phase_global = (self._phase_global + BASE_FREQ * self.dt) % 1.0
        if self._hold > 0:
            self._delta[:] = 0.0
        else:
            self._delta += FREQ_RANGE * freq_act * self.dt          # accumulate at the commanded rate
            self._delta *= (1.0 - RECENTER * self._relax)           # recenter toward trot when calm
            # Run 27: clamp WIDENS with off-balance d (= 1 - relax). Calm -> DELTA_MAX (clean trot);
            # off-balance -> up to DELTA_MAX_PUSH, so a leg can retime far enough to step under the COM.
            d_now = 1.0 - self._relax
            delta_max_eff = DELTA_MAX + (DELTA_MAX_PUSH - DELTA_MAX) * d_now
            self._delta = np.clip(self._delta, -delta_max_eff, delta_max_eff)  # d-gated coordination bound
        self._phase = (self._phase_global + TROT_OFFSET + self._delta) % 1.0

        holding = self._hold > 0
        if holding:
            self._hold -= 1
            q_target = self._default_joint.copy()                 # settle at the default stance
        else:
            # Run 27: FADE the FTG by relax (= 1 - d). Calm -> full FTG arc (clean trot unchanged);
            # off-balance -> the forced swing arc shrinks toward 0, handing the residual full authority
            # over the legs to brace / step for recovery instead of being yanked through a swing on clock.
            q_target = (self._default_joint
                        + self._relax * _ftg_offset(self._phase)
                        + RESIDUAL_SCALE * joint_res)

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
        z = self._local_height()

        vel_track = np.exp(-((vx - self._v_target) ** 2) / 0.25)

        contact = self._feet_contact()
        self._feet_air_time += self.dt
        first_contact = contact & (~self._last_contact)
        # Peaked airtime: at each touchdown, score how close that leg's flight time was to the shared
        # target (Gaussian). Same target for all legs -> pulls them to a common swing duration (symmetry).
        air_time_reward = float(np.sum(
            first_contact * np.exp(-((self._feet_air_time - AIR_TIME_TARGET) ** 2) / (2 * AIR_TOL ** 2))))
        self._feet_air_time *= ~contact
        self._last_contact = contact

        # Phase consistency (auxiliary): each leg in stance during the stance half of its phase,
        # swinging in the swing half. Coordination is structural now; this just keeps contacts honest.
        expected_stance = self._phase < 0.5
        phase_match = float(np.mean(np.where(expected_stance, contact, ~contact)))
        phase_reward = 2.0 * phase_match - 1.0

        foot_pos = self.data.geom_xpos[self._foot_geom_ids]
        foot_xy_speed = np.linalg.norm((foot_pos[:, :2] - self._prev_foot_xy) / self.dt, axis=1)
        self._prev_foot_xy = foot_pos[:, :2].copy()
        foot_clear = np.clip(foot_pos[:, 2] - self._foot_radius, 0.0, None)
        swing = ~contact
        clearance_reward = float(np.sum(
            swing
            * np.clip(foot_clear / FOOT_CLEAR_TARGET, 0.0, 1.0)
            * np.clip(foot_xy_speed / FOOT_CLEAR_VREF, 0.0, 1.0)
        ))

        # FTG-tracking shortfall penalty: each leg should reach the FTG's nominal foot clearance during
        # its swing; penalize only the SHORTFALL (foot below the nominal arc), never extra lift. Forces
        # every leg (incl. the propulsive rears) to follow the FTG instead of cancelling it, and is
        # symmetric across legs. d-gated below so terrain/recovery can deviate.
        ftg_lift = np.where(self._phase < 0.5, 0.0,
                            np.sin(np.pi * np.clip((self._phase - 0.5) / 0.5, 0.0, 1.0)))
        nominal_clear = FTG_FOOT_LIFT * ftg_lift
        track_shortfall = float(np.sum(np.clip(nominal_clear - foot_clear, 0.0, None)))
        swing_clear = foot_clear * swing            # per-leg clearance while swinging (0 in stance), logged

        if holding:
            vel_track = 0.0
            air_time_reward = 0.0
            clearance_reward = 0.0
            phase_reward = 0.0
            track_shortfall = 0.0

        orient_pen = 1.0 - up
        ang_vel_rp = self.data.qvel[3] ** 2 + self.data.qvel[4] ** 2
        yaw_rate = self.data.qvel[5] ** 2
        action_rate = np.sum((action - self._prev_action) ** 2)
        torque_cost = np.sum(self.data.ctrl ** 2)
        joint_vel = np.sum(self.data.qvel[6:] ** 2)
        pose_dev = np.sum((self.data.qpos[7:] - self._default_joint) ** 2)
        abduct_dev = np.sum((self.data.qpos[7:][self._abduct_idx]
                             - self._default_joint[self._abduct_idx]) ** 2)
        freq_dev = float(np.sum(freq_act ** 2))

        tilt = 1.0 - up
        rate = np.sqrt(ang_vel_rp)
        d_off = float(np.clip(max((tilt - D_TILT_DEAD) / D_TILT_REF,
                                  (rate - D_RATE_DEAD) / D_RATE_REF), 0.0, 1.0))
        relax = 1.0 - d_off
        self._relax = relax          # gate next step's phase-offset RECENTER (calm -> recenter to trot)
        target_h = TARGET_HEIGHT - HEIGHT_DROP_MAX * d_off
        height_track = np.exp(-((z - target_h) ** 2) / (2 * HEIGHT_TOL ** 2))

        reward = (
            W_VEL * vel_track             # Run 24: 1.5 -> 1.0 (W_VEL); velocity was dominating the gradient
            + W_HEIGHT * height_track     # Run 24: 0.8 -> 0.5 (keep ~2:1 vel:height so it still walks)
            + 0.1
            + W_AIRTIME * air_time_reward   # Run 25: restored as peaked target (swing-duration symmetry)
            + W_CLEARANCE * clearance_reward
            + relax * W_PHASE * phase_reward
            - 0.5 * orient_pen
            - W_ANGVEL * ang_vel_rp
            - W_YAW * yaw_rate
            - W_ACTRATE * action_rate
            - 1e-5 * torque_cost
            - W_JOINTVEL * joint_vel
            - relax * W_POSE * pose_dev   # Run 21: restored (regularizes residual thrash; now refs the taller nominal)
            - relax * W_ABDUCT * abduct_dev
            - relax * W_FREQ * freq_dev
            - relax * W_TRACK * track_shortfall   # Run 23: make every leg follow the FTG swing (kills rear-leg flatten)
        )   # Run 20: dropped r_airtime (redundant w/ FTG). r_pose restored Run 21.

        self._prev_action = action.copy()
        terminated = bool(z < 0.18 or up < 0.4) and not holding

        info = {
            "vx": float(vx), "v_target": float(self._v_target), "z": float(z),
            "r_vel": float(W_VEL * vel_track),
            "r_height": float(W_HEIGHT * height_track),
            "r_alive": 0.1,
            "r_airtime": float(W_AIRTIME * air_time_reward),
            "r_clearance": float(W_CLEARANCE * clearance_reward),
            "r_phase": float(relax * W_PHASE * phase_reward),
            "r_orient": float(-0.5 * orient_pen),
            "r_angvel": float(-W_ANGVEL * ang_vel_rp),
            "r_yaw": float(-W_YAW * yaw_rate),
            "r_action_rate": float(-W_ACTRATE * action_rate),
            "r_torque": float(-1e-5 * torque_cost),
            "r_jointvel": float(-W_JOINTVEL * joint_vel),
            "r_pose": float(-relax * W_POSE * pose_dev),
            "r_abduct": float(-relax * W_ABDUCT * abduct_dev),
            "r_freq": float(-relax * W_FREQ * freq_dev),
            "r_track": float(-relax * W_TRACK * track_shortfall),
            # per-leg swing clearance (m) + swing flags, for the breakdown's per-leg lift diagnostic:
            "swc_FL": float(swing_clear[0]), "swc_FR": float(swing_clear[1]),
            "swc_RL": float(swing_clear[2]), "swc_RR": float(swing_clear[3]),
            "sw_FL": float(swing[0]), "sw_FR": float(swing[1]),
            "sw_RL": float(swing[2]), "sw_RR": float(swing[3]),
            # per-leg duty factor (fraction of time in stance/contact) -> high = propulsive support leg:
            "dut_FL": float(contact[0]), "dut_FR": float(contact[1]),
            "dut_RL": float(contact[2]), "dut_RR": float(contact[3]),
        }
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, False, info

    def reset_model(self):
        if self._terrain:
            # Randomize box-tile heights via mocap_pos z (tile top = mocap_z + half_z). Fractal terrain
            # computed from each tile's (x,y) centre; mocap bodies -> collision tracks the new height.
            xy = self.data.mocap_pos[self._tile_mocap, :2]
            H = _terrain_heights(xy, self.np_random, self._terrain_height)
            self.data.mocap_pos[self._tile_mocap, 2] = H - self._tile_half_z
            self.model.geom_friction[self._foot_geom_ids, 0] = (
                self._foot_friction0 * self.np_random.uniform(0.6, 1.4, size=len(self._foot_geom_ids)))

        qpos = self.model.key_qpos[0].copy() if self.model.nkey > 0 else self.init_qpos.copy()
        qpos += self.np_random.uniform(-0.02, 0.02, self.model.nq)
        qpos[2] += self._spawn_lift
        qvel = self.init_qvel + self.np_random.uniform(-0.02, 0.02, self.model.nv)
        self.set_state(qpos, qvel)
        self.data.ctrl[:] = 0.0

        self._v_target = self.np_random.uniform(0.5, 1.0)
        self._prev_action = np.zeros(self._n_joints + N_LEGS)
        self._feet_air_time = np.zeros(4)
        self._last_contact = np.ones(4, dtype=bool)
        self._prev_foot_xy = self.data.geom_xpos[self._foot_geom_ids, :2].copy()
        # Shared clock starts at a random rotation; per-leg offsets start at 0, so phases begin at the
        # FIXED trot offsets (the diagonal RELATIONSHIP is preserved exactly every episode).
        self._phase_global = float(self.np_random.uniform(0.0, 1.0))
        self._delta = np.zeros(N_LEGS)
        self._relax = 1.0
        self._phase = (self._phase_global + TROT_OFFSET) % 1.0
        self._hold = self._hold_steps
        self._push_force[:] = 0.0
        self._push_left = 0
        self._push_active = False
        self._push_cooldown = int(self.np_random.integers(self._push_int_min, self._push_int_max + 1))
        if self._push:
            self.data.xfrc_applied[self._base_id, :3] = 0.0
        self._local_height()
        return self._get_obs()


def make_env(**kw):
    return Go2Env(**kw)


# Reward components in display order (must match the keys put in info dict in step()).
REWARD_KEYS = ["r_vel", "r_height", "r_alive", "r_airtime", "r_clearance", "r_phase",
               "r_orient", "r_angvel", "r_yaw", "r_action_rate", "r_torque", "r_jointvel",
               "r_pose", "r_abduct", "r_freq", "r_track"]

# Per-leg swing-clearance diagnostic: swc_* sums foot clearance while that leg swings, sw_* counts the
# swing steps -> mean swing clearance = swc/sw per leg. Reveals legs (e.g. the rears) that barely lift.
SWING_CLEAR_KEYS = ["swc_FL", "swc_FR", "swc_RL", "swc_RR"]
SWING_FLAG_KEYS = ["sw_FL", "sw_FR", "sw_RL", "sw_RR"]
# Per-leg duty factor: dut_* sums contact (1 in stance, 0 in swing); mean over steps = fraction of the
# cycle that leg is on the ground. A leg with high duty + low clearance = a propulsive support leg
# (validates "front-high / rear-low" as a real push gait rather than just a lazy non-lift).
DUTY_KEYS = ["dut_FL", "dut_FR", "dut_RL", "dut_RR"]


def format_reward_breakdown(total_steps, window_steps, sums, count, vx_sum, z_sum,
                            swc_sums=None, sw_sums=None, duty_sums=None):
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
    if swc_sums is not None and sw_sums is not None:
        out.append("")
        out.append("per-leg mean SWING clearance (m)  [rear << front = rears not lifting]:")
        for leg, ck, fk in zip(["FL", "FR", "RL", "RR"], SWING_CLEAR_KEYS, SWING_FLAG_KEYS):
            sw = sw_sums.get(fk, 0.0)
            mean_c = (swc_sums.get(ck, 0.0) / sw) if sw > 0 else 0.0
            out.append(f"  {leg} : {mean_c:.4f}")
    if duty_sums is not None:
        out.append("")
        out.append("per-leg DUTY factor (stance fraction)  [rear >> front = rears propelling]:")
        for leg, dk in zip(["FL", "FR", "RL", "RR"], DUTY_KEYS):
            out.append(f"  {leg} : {duty_sums.get(dk, 0.0) / count:.3f}")
    return "\n".join(out) + "\n"


class RewardBreakdownCallback(BaseCallback):
    """Accumulates per-term rewards and writes a .txt breakdown at each checkpoint interval."""
    def __init__(self, save_freq, save_path, name_prefix="ppo_go2"):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self._reset()

    def _reset(self):
        self.sums = {k: 0.0 for k in REWARD_KEYS}
        self.swc_sums = {k: 0.0 for k in SWING_CLEAR_KEYS}
        self.sw_sums = {k: 0.0 for k in SWING_FLAG_KEYS}
        self.duty_sums = {k: 0.0 for k in DUTY_KEYS}
        self.vx_sum = 0.0
        self.z_sum = 0.0
        self.count = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "r_vel" not in info:
                continue
            for k in REWARD_KEYS:
                self.sums[k] += info.get(k, 0.0)
            for k in SWING_CLEAR_KEYS:
                self.swc_sums[k] += info.get(k, 0.0)
            for k in SWING_FLAG_KEYS:
                self.sw_sums[k] += info.get(k, 0.0)
            for k in DUTY_KEYS:
                self.duty_sums[k] += info.get(k, 0.0)
            self.vx_sum += info.get("vx", 0.0)
            self.z_sum += info.get("z", 0.0)
            self.count += 1
        if self.n_calls % self.save_freq == 0 and self.count > 0:
            os.makedirs(self.save_path, exist_ok=True)
            text = format_reward_breakdown(
                self.num_timesteps, self.save_freq * len(self.locals.get("infos", [1])),
                self.sums, self.count, self.vx_sum, self.z_sum,
                self.swc_sums, self.sw_sums, self.duty_sums,
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
    parser = argparse.ArgumentParser(description="Train the Go2 walking policy (Tier-3 PMTG/FTG).")
    parser.add_argument("--terrain", action="store_true",
                        help="train on uneven box-tile terrain instead of flat ground")
    parser.add_argument("--terrain-height", type=float, default=0.15,
                        help="max terrain elevation in metres (default 0.15; curriculum up later)")
    parser.add_argument("--push", action="store_true",
                        help="enable push/disturbance training (random torso shoves) + warm-start "
                             "exploration boost, so the policy learns to recover from impulses")
    args = parser.parse_args()

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
    print(f"[tier3] PMTG/FTG: shared base clock {BASE_FREQ} Hz + trot offsets {TROT_OFFSET.tolist()}; "
          f"FTG swing (thigh {FTG_THIGH_AMP}, calf {FTG_CALF_AMP}) + residual x{RESIDUAL_SCALE}; "
          f"per-leg offset clamp +/-{DELTA_MAX} (recenter {RECENTER}); "
          f"action={venv.action_space.shape[0]}-D, obs={venv.observation_space.shape[0]}-D")

    if WARM_START:
        model_path, vec_path = WARM_START
        venv = VecNormalize.load(vec_path, venv)
        venv.training = True
        venv.norm_reward = True
        model = PPO.load(model_path, env=venv, device="auto", tensorboard_log="runs")
        print(f"[warm-start] loaded {model_path}")
        if args.terrain or args.push:
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
        save_freq=save_freq, save_path=CKPT_DIR, name_prefix="ppo_go2", save_vecnormalize=True,
    )
    breakdown = RewardBreakdownCallback(save_freq=save_freq, save_path=CKPT_DIR, name_prefix="ppo_go2")
    callbacks = CallbackList([ckpt, breakdown])

    try:
        model.learn(total_timesteps=TOTAL_STEPS, callback=callbacks, progress_bar=True,
                    reset_num_timesteps=True)
    except KeyboardInterrupt:
        print("\n[interrupted] saving current model before exit...")
    finally:
        model.save(os.path.join(CKPT_DIR, "ppo_go2_0_steps"))
        venv.save(os.path.join(CKPT_DIR, "ppo_go2_vecnormalize_0_steps.pkl"))
        print(f"Saved latest model -> {CKPT_DIR}ppo_go2_0_steps.zip (+ vecnormalize). Watch it with: 0")
        if breakdown.count > 0:
            fn = os.path.join(CKPT_DIR, "ppo_go2_reward_breakdown_0_steps.txt")
            with open(fn, "w") as f:
                f.write(format_reward_breakdown(
                    model.num_timesteps, breakdown.count, breakdown.sums,
                    breakdown.count, breakdown.vx_sum, breakdown.z_sum,
                    breakdown.swc_sums, breakdown.sw_sums, breakdown.duty_sums))
            print(f"Saved reward breakdown -> {fn}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Tuning knobs for future runs (log each change as a new Run in report_logs_3.md):
#   FTG_THIGH_AMP / FTG_CALF_AMP -> nominal swing-arc height (bigger = higher step; ~0.09 m foot lift now)
#   RESIDUAL_SCALE -> how much the policy may deviate from the FTG nominal (smaller = closer to a pure trot)
#   DELTA_MAX -> hard bound on per-leg phase offset from trot (raise for terrain retiming; lower = tighter trot)
#   FREQ_RANGE -> how fast the freq action moves a leg's offset; RECENTER -> pull back to trot when calm
#   W_FREQ    -> small penalty on clock churn (DELTA_MAX/RECENTER do the real coordination work)
#   W_PHASE / W_CLEARANCE        -> auxiliary now (the FTG handles arc + coordination)
# ---------------------------------------------------------------------------
