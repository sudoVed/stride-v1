# STRIDE — Quadruped RL Training Log (Flat / No Terrain)

Reinforcement-learning locomotion for the **Unitree Go2** quadruped in **MuJoCo**, trained with
**PPO** to walk forward on **flat ground**. This file is the research record: it documents the fixed
training setup, every input/output and reward term (and the run each was introduced in), and then a
per-run history of what changed, what went wrong, and the fix that followed.

**Stack:** MuJoCo 3.9.0 · `mujoco` (Python) · Gymnasium 1.3.0 · Stable-Baselines3 2.8.0 · PyTorch ·
Python 3.12.10 · native Windows. **Script:** `train_legacy.py` (the flat-era 12-D script, now under
`python/legacy/legacy_12D_output/`) · **Viewer:** `watch_legacy.py` (native MuJoCo viewer).

> **STATUS — flat locomotion SOLVED (Runs 1–10).** Run 8 is the reference policy: a clean, converged
> diagonal trot (no splay, tilt, or spin). Run 10 showed the trot is **self-sustaining** without the
> gait-schedule reward, so the schedule was a learning crutch, not a permanent requirement. The final
> reward weights are in §1.5 below. Work continues on **uneven terrain** in `report_logs_2.md`,
> warm-started from the Run 8 model.

---

# 1 · Training Setup & Reference

This section is the current state of the system. Per-run deltas are in Section 2; the "since" column
notes the run a value was introduced or last changed.

## 1.1 Task & environment (`Go2Env`)

| Property | Value | Since |
|---|---|---|
| Model | Unitree Go2 (`mujoco_menagerie/unitree_go2/scene.xml`) | Run 1 |
| Control rate | 50 Hz (`frame_skip=10`, physics timestep 0.002 s) | Run 1 |
| Control scheme | **PD position control**: `τ = Kp·(q_target − q) − Kd·q̇`, recomputed every physics substep | Run 3 |
| PD gains | `Kp = 30`, `Kd = 0.75` | Run 3 |
| Action mapping | `q_target = default_pose + ACTION_SCALE · action`, `ACTION_SCALE = 0.7` | Run 3 (scale 0.5→0.7 in Run 4) |
| Reset | `home` keyframe + `U(−0.02, 0.02)` noise on qpos/qvel | Run 2 |
| Speed command | `v_target ~ U[0.5, 1.0]` m/s, resampled per episode | Run 2 |
| Termination | torso height < 0.18 m **or** `torso_up < 0.4` (~66° tilt) | Run 2 |
| Episode limit | 1000 steps (`TimeLimit`) | Run 2 |
| Normalization | `VecNormalize` (obs + reward, `clip_obs=10`), 8 parallel envs | Run 1 |

Reachable torso-height ceiling (offline kinematics): 0.34 m @ ACTION_SCALE 0.5, 0.36 m @ 0.7,
~0.39 m absolute (knees at limit — a stiff locked posture to avoid; calf limit −0.838 rad).

## 1.2 PPO hyperparameters (unchanged since Run 1)

| policy | n_envs | n_steps | batch | n_epochs | gamma | gae_λ | lr | clip | ent_coef | total steps |
|---|---|---|---|---|---|---|---|---|---|---|
| MlpPolicy `[256,256]` | 8 | 2048 | 4096 | 10 | 0.99 | 0.95 | 3e-4 | 0.2 | 0.0 | 3,000,000 |

## 1.3 Inputs — observation vector (50-D)

Pure proprioception; no vision. Normalized by `VecNormalize`.

| Component | Dim | Since |
|---|---|---|
| Torso height (`qpos[2]`) | 1 | Run 1 |
| Orientation quaternion | 4 | Run 1 |
| Joint angles | 12 | Run 1 |
| Base linear velocity | 3 | Run 1 |
| Base angular velocity | 3 | Run 1 |
| Joint velocities | 12 | Run 1 |
| Commanded speed `v_target` | 1 | Run 2 |
| Previous action | 12 | Run 2 |
| Gait-phase clock `sin, cos` | 2 | Run 6 |

**Dimension history:** 35 (Run 1) → 48 (Run 2, +command +prev-action) → 50 (Run 6, +phase).

## 1.4 Output — action vector (12-D)

One value per joint, each in `[−1, 1]`. **Not torque, not absolute angle** — a normalized *offset
from the default standing pose*; the PD law (§1.1) converts it to torque each substep. During
training the `MlpPolicy` emits a Gaussian (mean + learned std) per joint and samples it; the viewers
use the deterministic mean. A separate **critic head** outputs a scalar value estimate (training
only). Action representation: raw torque in Runs 1–2, **PD offset since Run 3**.

## 1.5 Reward terms (current set)

Per-step reward is the signed sum of the terms below; each is logged to the per-checkpoint
breakdown file. Positives first, then penalties.

| Term | Formula (weight) | Intent | Since (changes) |
|---|---|---|---|
| `r_vel` | `1.5 · exp(−(vx−v_target)²/0.25)` | track commanded forward speed | Run 1 (raw speed) → Run 2 (tracking form) |
| `r_height` | `0.5 · exp(−(z−0.33)²/(2·0.05²))` | hold torso at target height | Run 4 (target 0.32) → Run 6 (0.33) |
| `r_alive` | `+0.1` | small bonus for staying up | Run 1 (0.5) → Run 2 (0.1) |
| `r_airtime` | `0.5 · Σ(air_time − 0.2)` at touchdown | reward real, lifted steps | Run 2 (thr 0.3) → Run 5 (0.2) |
| `r_clearance` | `0.10 · Σ_swing clip(clear/0.08)·clip(v_xy/0.30)` | lift swing feet (speed-gated) | Run 5 (0.30) → Run 7 (0.10) |
| `r_gait` | `W_GAIT · (2·match − 1)`, `W_GAIT = 0` | match diagonal-trot schedule | Run 6 (0.5) → Run 9 (0.0) |
| `r_orient` | `−0.5 · (1 − up)` | keep torso level (tilt magnitude) | Run 2 |
| `r_angvel` | `−0.05 · (ω_roll² + ω_pitch²)` | damp torso buck (pitch/roll *rate*) | Run 7 |
| `r_yaw` | `−0.15 · ω_yaw²` | stop body turning/twisting | Run 8 |
| `r_action_rate` | `−3e-4 · Σ Δaction²` | smooth control | Run 2 (1e-4) → Run 7 (3e-4) |
| `r_torque` | `−1e-5 · Σ τ²` | energy / applied-torque cost | Run 2 (energy) → PD form since Run 3 |
| `r_jointvel` | `−1.5e-3 · Σ q̇²` | anti-thrash / anti-overmovement | Run 2 (5e-4) → Run 7 (1.5e-3) |
| `r_pose` | `−0.02 · Σ(q − q_default)²` | pull toward default symmetric stance | Run 5 |
| `r_abduct` | `−0.5 · Σ_hip(q − q_default)²` | hold legs under body (no outward splay) | Run 8 |

**Gait schedule (`r_gait`) detail:** a phase clock advances at `GAIT_FREQ = 1.5` cycles/s and is fed
to the policy as `sin/cos` (§1.3). Diagonal pair A = FL+RR is scheduled in stance for the first half
of each cycle while B = FR+RL swings, then they swap. `match` = fraction of feet whose contact agrees
with the schedule; the `2·match − 1` centering makes a correct in-phase trot score +1, while
standing / hopping / pacing score 0 and anti-phase scores −1. `W_GAIT = 0` currently (disabled for
the emergent-gait experiment, Runs 9–10).

## 1.6 Instrumentation & workflow

| Feature | Description | Since |
|---|---|---|
| Per-checkpoint breakdown | `ppo_go2_reward_breakdown_<N>_steps.txt`: every term's mean/step + share, mean vx, mean z | Run 4 |
| Checkpoints | every 750k steps (`CheckpointCallback`, + VecNormalize `.pkl`) | Run 2 |
| Ctrl+C / final save | written into checkpoints as step `0` (`ppo_go2_0_steps`) → watchable with arg `0` | Run 8 |
| Checkpoint wipe | `checkpoints/` is `rmtree`'d at the start of every run (no stale-file misreads) | Run 8 |
| Warm-start | `WARM_START = (model.zip, vecnorm.pkl)` continues from a saved policy; `None` = scratch | Run 10 |

---

# 2 · Run History

Each run: **metadata → changed from previous → problem analysis → proposed solution.** "SUCCESS"
marks a run that met its stated goal; the reference policy is **Run 8**.

---

## Run 1 — Baseline reward

**Date:** 2026-06-13 · **Model:** Go2 · **Algorithm:** PPO `MlpPolicy` · **Status:** stopped ~753k/3M for analysis · **Result:** reward-hacked standing.

**Changed from previous:** first run. Reward = `1.5·forward_vel + 0.5·alive − 5e-4·Σaction²`. Obs 35-D,
position-target action, **no episode time limit**, termination only on torso height ≤ 0.18 m.

**Problem analysis — reward hacking via the alive bonus.** `ep_len_mean` 6610, `ep_rew_mean` 3250.
Arithmetic: `0.5 alive × 6610 = 3305 ≈ ep_rew` → **~100 % of return is the flat alive bonus**; the
velocity term contributes ~nothing. The policy stands still and farms the bonus. Two faults: (1) with
no time limit and fall-only termination, "stay up and do nothing" has unbounded return; (2) the +0.5
alive bonus out-competes the risk of walking. Optimizer is healthy (`explained_variance` 0.90, small
`approx_kl`) — the **reward spec**, not the algorithm, is at fault.

**Proposed solution → Run 2.** Add a 1000-step `TimeLimit`; shrink the alive bonus; redesign the
reward around velocity *tracking* plus locomotion shaping (orientation, action-rate, joint-velocity,
foot air-time); add tilt-based early termination.

---

## Run 2 — Reward redesign + time limit

**Date:** 2026-06-13 · **Status:** stopped ~688k/3M · **Result:** faceplants in ~0.3 s.

**Changed from previous:** added `max_episode_steps = 1000`; alive 0.5→0.1; reward redesigned to
velocity tracking + air-time + orientation/action-rate/joint-velocity penalties; obs 35→48 (added
`v_target` + previous action); spawn from `home` keyframe; **confirmed actuators are raw torque
motors** (action = 12 torques); checkpoints every 750k.

**Problem analysis — collapses immediately (action representation).** `ep_len_mean` 16.5 → ~0.33 s
before falling; `ep_rew_mean` 4.97 ≈ 0.30/step (just the alive bonus before toppling), flat over
688k. The reward exploit is fixed but the **action representation** is wrong: with raw torque motors,
near-zero output resists nothing, so merely standing requires the policy to compute
gravity-compensating torque at all 12 joints every step before it can walk. From random init it
topples by step ~16 and never accumulates upright experience. `clip_fraction` 0.28 / `approx_kl` 0.02
corroborate thrashing. **Offline check:** zero-torque model falls by control step 12 (~0.24 s,
matching `ep_len`); under PD position control commanding the default pose it stands stably 6 s.

**Proposed solution → Run 3.** Adopt **PD position control**: policy outputs a normalized offset
(action ∈ [−1,1]) around the default pose; PD law converts it to torque each substep. Commanding the
default pose auto-produces restoring torque, so the robot holds itself up and only learns the deltas.

---

## Run 3 — PD position control

**Date:** 2026-06-13 · **Status:** SUCCESS (first walking policy) · **Result:** walks, but in a low crouch.

**Changed from previous:** PD position control replaces raw torque (Kp 30, Kd 0.75,
ACTION_SCALE 0.5); action space redefined to `Box(−1,1,12)`. Reward, time limit, obs unchanged.

**Problem analysis — phase-change to a walking gait.** `ep_len_mean` jumped 16 → ~1000 as a step
change (survival is binary: fall vs the 1000-step cap); once PD kept it upright, every episode ran to
the cap. `ep_rew_mean ≈ 1120` ≈ 1.12/step × 1000. A stationary policy would score only ~0.25/step
(alive 0.1 + standstill velocity-track ≈ 0.16); observed 1.12/step ⇒ `vel_track ≈ 0.6` ⇒ genuine
forward locomotion within ~0.35 m/s of target. Training healthy (`approx_kl` 0.012, `expl_var` 0.98),
`std` 0.83 (still improving). **Visual:** walks in a **low, knees-bent crouch** — a low CoM is the
most stable way to satisfy the reward, and nothing yet rewards height.

**Proposed solution → Run 4.** Add a torso-height **tracking** reward (Gaussian peak, ungameable by
hopping) and raise ACTION_SCALE so the target is reachable.

---

## Run 4 — Height tracking

**Date:** 2026-06-13 · **Status:** SUCCESS (height goal) · **Result:** stands up (0.26→0.31 m), but shuffles.

**Changed from previous:** new `r_height = 0.5·exp(−(z−0.32)²/(2·0.05²))` (target 0.32 m, ±0.05);
ACTION_SCALE 0.5→0.7. **Instrumentation added:** per-checkpoint reward-breakdown file.

**Key metrics (final, 3.0M):** `ep_rew` 1970, `std` 0.48, `expl_var` 0.925.

| term | 750k | 1.5M | 2.25M | 3.0M |
|------|-----:|-----:|------:|-----:|
| r_vel | 0.794 | 1.425 | 1.472 | 1.484 |
| r_height | 0.246 | 0.383 | 0.453 | 0.481 |
| r_airtime | -0.037 | -0.043 | -0.036 | -0.027 |
| r_jointvel | -0.085 | -0.084 | -0.076 | -0.066 |
| **TOTAL** | 0.985 | 1.757 | 1.891 | **1.951** |
| mean vx | 0.323 | 0.734 | 0.743 | 0.744 |
| mean z | 0.259 | 0.285 | 0.300 | **0.309** |

**Problem analysis — height met, gait is a low-effort shuffle.** TOTAL 1.951 × 1000 ≈ `ep_rew` 1970
(breakdown reconciles). Height climbed to **0.309 m** (`r_height` 0.481 ⇒ |z−0.32| ≈ 0.013) — crouch
solved. Velocity perfect (`r_vel` 1.484 ⇒ vel_track 0.99, vx 0.744). **But `r_airtime` stays negative
all run** (−0.027): the feet that touch down were airborne < 0.3 s (short shuffly steps) and the hind
feet barely lift — the quantitative fingerprint of the visual shuffle. It *sacrifices* the air-time
reward rather than earn it. Converged (`std` 0.48). No term constrains per-leg symmetry → front/hind
asymmetry.

**Proposed solution → Run 5.** Make stepping pay: lower the air-time threshold (0.3→0.2) and add an
explicit **swing-foot clearance** reward. (Reserved: default-pose regularization for symmetry.)

---

## Run 5 — Clearance + pose regularization

**Date:** 2026-06-13 · **Status:** stopped ~1.54M/3M · **Result:** lifts feet, but uncoordinated "zombie hop".

**Changed from previous:** new `r_clearance = 0.30·Σ_swing clip(clear/0.08)·clip(v_xy/0.30)` (lift
swing feet, speed-gated); air-time threshold 0.3→0.2; new `r_pose = −0.02·Σ(q−q_default)²`.

**Key metrics (1.5M):** `ep_rew` 2160, `ep_len` 972 (some falls), `std` 0.755.

| term | 750k | 1.5M |
|------|-----:|-----:|
| r_clearance | 0.163 | 0.304 |
| r_airtime | -0.016 | -0.007 |
| r_height | 0.251 | 0.375 |
| r_orient | -0.015 | -0.009 |
| r_jointvel | -0.088 | -0.100 |

**Problem analysis — it lifts, but doesn't walk.** `r_clearance` +0.30 and `r_airtime` ≈ 0 → feet
leave the ground, but lifting without coordination isn't a gait: (1) it crouched to hop — `r_height`
0.375 ⇒ z ≈ **0.28 m**, below Run 4; (2) torso rocks — `r_orient` −0.009 ⇒ **~11° tilt**; (3) more
thrash — `r_jointvel` −0.066→−0.100, `ep_len` 1000→972. Root cause: the per-foot terms say feet
should lift, but **not *which* feet, *when*, or *in what phase*** — the reward is coordination-blind,
so the optimizer found a flailing hop.

**Proposed solution → Run 6.** Stop hoping a gait emerges and **prescribe one**: a diagonal-trot
contact schedule (phase clock in obs; reward matching FL+RR / FR+RL alternation). Also raise height
target 0.32→0.33.

---

## Run 6 — Diagonal-trot schedule

**Date:** 2026-06-13 · **Status:** ran to 3M · **Result:** trot locks in, but bucks wildly.

**Changed from previous:** new `r_gait = 0.5·(2·match−1)` (diagonal contact schedule); gait phase
added to obs (48→50-D); height target 0.32→0.33.

**Key metrics (3.0M):** `ep_rew` 2900, `std` 0.476, `expl_var` 0.81.

| term | 1.5M | 3.0M |
|------|-----:|-----:|
| r_gait | 0.300 | 0.377 |
| r_clearance | 0.430 | **0.612** |
| r_jointvel | -0.089 | -0.083 |
| r_orient | -0.012 | -0.015 |
| mean z | 0.279 | 0.305 |

**Problem analysis — the schedule worked, and that unleashed the bucking.** Trot locked in: `r_gait`
0.377 ⇒ **~88 % schedule match**; height recovered to 0.305; velocity perfect. **But** `r_clearance`
ran away 0.43→**0.61** (saturated, ~2.0/2.0 for the swing feet): with the schedule now supplying
coordination, the strong clearance reward is redundant and just pays the policy to fling its legs
high/fast — the joint overmovement. The restraining penalties can't keep up (`r_jointvel` −0.083 vs
+0.61 clearance), and **nothing penalizes the buck itself** — `r_orient` sees tilt *magnitude*, not
the *rate* of pitch/roll, so rapid bucking is nearly free.

**Proposed solution → Run 7.** Cut clearance 0.30→0.10; add a roll+pitch **angular-velocity penalty**
(`r_angvel`, the missing anti-buck term); raise joint-velocity (5e-4→1.5e-3) and action-rate
(1e-4→3e-4) penalties to bound overmovement.

---

## Run 7 — Clearance cut + anti-buck penalties

**Date:** 2026-06-13 · **Status:** stopped ~2.25M/3M · **Result:** much calmer; residual yaw-spin + leg splay.

**Changed from previous:** `W_CLEARANCE` 0.30→0.10; new `r_angvel = −0.05·(ω_roll²+ω_pitch²)`;
`r_jointvel` 5e-4→1.5e-3; `r_action_rate` 1e-4→3e-4.

**Key metrics (2.25M):** mean vx 0.728, mean z 0.307.

| term | value | share |
|------|------:|------:|
| r_vel | 1.469 | 67.1% |
| r_gait | 0.401 | 18.3% |
| r_clearance | 0.114 | 5.2% |
| r_angvel | -0.101 | -4.6% |
| r_jointvel | -0.197 | -9.0% |
| r_orient | -0.003 | -0.1% |

**Problem analysis — cleanup worked; remaining faults are un-penalized DOFs.** Clearance tamed
(0.61→**0.114**); bucking down (`r_orient` ⇒ **~6° tilt**, was 14°; `r_angvel` actively damping); trot
tighter (`r_gait` ⇒ **~90 % match**). Three faults remain, each an unconstrained DOF: (1) **lateral
body twisting = yaw**, and `r_angvel` covers roll+pitch only — **yaw (`qvel[5]`) has no term**; (2)
**legs splay outward** — the `*_hip_joint` abduction DOFs are only weakly held by the all-joints
`r_pose` (0.02); (3) FL overextends (asymmetry, no symmetry term).

**Proposed solution → Run 8.** Two penalty-only additions: a **yaw-rate penalty** (`r_yaw`) to stop
turning, and a **hip-abduction penalty** (`r_abduct`) to hold legs under the body. (Action scale left
uniform — capping it would mask the splay, not cure it.)

---

## Run 8 — Yaw + abduction penalties ✅ reference policy

**Date:** 2026-06-13 · **Status:** ran to 3M, **converged** · **Result:** clean diagonal trot — no splay, no tilt, no spin.

**Changed from previous:** new `r_yaw = −0.15·ω_yaw²`; new `r_abduct = −0.5·Σ_hip(q−q_default)²`.
Workflow: checkpoint-wipe at start of run; step-`0` save on Ctrl+C.

**★ Preserved checkpoint (key POI):** `python/legacy/legacy_12D_output/run8_gait/` (3M; was `checkpoints-final-gait`).
This is the clean-trot reference policy and the warm-start source for the entire robustness line
(Runs 13–15). Worth keeping — it's the canonical "flat walking solved" milestone.

**Key metrics (3.0M):** `ep_rew` 2330, `std` **0.352**, `entropy_loss` −4.29, `expl_var` 0.967, mean vx 0.741, mean z 0.310.

| term | Run 7 @2.25M | Run 8 @3.0M |
|------|------:|------:|
| r_vel | 1.469 | 1.481 |
| r_gait | 0.401 | 0.423 |
| r_orient | -0.003 | -0.0015 |
| r_angvel | -0.101 | -0.067 |
| r_yaw | — | -0.013 |
| r_abduct | — | -0.024 |
| r_jointvel | -0.197 | -0.170 |
| **TOTAL** | 2.189 | **2.274** |

**Problem analysis — this is the one: clean, converged trot.** Every observed fix shows in the
metrics: **no spin** (`r_yaw` −0.013 ⇒ ~0.29 rad/s yaw, previously un-penalized); **no splay** (`r_abduct`
−0.024 ⇒ hip deviation ≈ 0.048, ~0.11 rad/hip — legs sagittal-plane); **no tilt** (`r_orient` ⇒ **~4°**,
was 14°; `r_angvel` 2.03→1.33); **tight trot** (`r_gait` ⇒ **~92 % match**, 88→90→92 across Runs 6–8;
`r_jointvel` raw 166→131→113 — calmer each run); velocity/height held. **Converged:** `std`
0.476→0.352, `entropy` −8.08→−4.29 — committed to a near-deterministic gait. Runs 1→8 went from
reward-hacked standing, through a faceplant, crouch-walk, shuffle, zombie-hop, and bucking-splay
trot, to a stable diagonal trot — each step fixing the single quantity the reward left unconstrained.

**Proposed solution → Run 9 (experiment).** With shaping now "right," remove the gait restriction
(`W_GAIT 0.5→0`) to see what gait emerges under good rewards/penalties alone.

---

## Run 9 — Gait reward off (emergent-gait experiment)

**Date:** 2026-06-13 · **Status:** stopped ~1.5M/3M (from scratch) · **Result:** shuffles — no gait emerges.

**Changed from previous:** `W_GAIT` 0.5→0 (term stays wired, contributes nothing — reversible, no
key/sum mismatch). Everything else unchanged. Trained **from random init**.

**Key metrics (1.5M):** mean vx 0.705, mean z 0.284.

| term | Run 8 (trot) | Run 9 (no schedule) |
|------|------:|------:|
| r_clearance | 0.113 | **0.025** |
| r_airtime | 0.006 | **−0.023** |
| r_angvel | -0.067 | **−0.165** |
| r_yaw | -0.013 | **−0.035** |
| r_jointvel | -0.170 | -0.221 |
| **TOTAL** | 2.274 | 1.359 |

**Problem analysis — no coordinated gait emerges under this reward/penalty scheme.** With the
schedule off, the policy collapses to a low-effort **shuffle** showing the exact Run-4 signature:
`r_clearance` 0.113→**0.025** and `r_airtime` **negative** (feet barely leave the ground, no swing
clears 0.2 s); height drops to 0.284. Tellingly, removing the schedule made it **less stable, not
freer**: `r_angvel` roughly doubled and `r_yaw` nearly tripled — the uncoordinated shuffle bucks and
twists *more* than the trot. So `r_gait` was doing real **coordination and stabilization** work, not
just imposing a style. **Conclusion: because of the current reward/penalty scheme, no gait arises on
its own** — velocity + height + a weak clearance term + the calm-motion penalties leave a
low-clearance shuffle as the easiest optimum, with no incentive that makes a phased trot beat it. The
Runs 6–8 trot was schedule-induced; the schedule was load-bearing.

**Proposed solution → Run 10.** Run 9 asked "does a trot *emerge* from scratch?" (no). Ask the better
question: **does the learned trot *persist*?** Warm-start from the Run 8 trot with `W_GAIT=0`.

---

## Run 10 — Warm-start from the trot (persistence test)

**Date:** 2026-06-13 · **Status:** ran to ~2.26M/3M · **Result:** SUCCESS — trot **persists and refines**; gait reward confirmed a learning crutch.

**Changed from previous:** **warm-start** — `WARM_START` loads the preserved Run 8 trot
(`run8_gait/ppo_go2_3000000_steps.zip` + its VecNormalize) and continues training
(`reset_num_timesteps=True`, so checkpoints renumber from 750k). `W_GAIT` stays 0. The *only*
difference from Run 9 is the starting point: a policy that already trots vs. a random one.

**Key metrics (2.25M):** `ep_rew` 2030, `std` **0.181**, `entropy_loss` +4.01 (very peaked policy), `expl_var` 0.984, mean vx 0.735, mean z 0.317.

| term | Run 8 (with schedule) | Run 9 (scratch, no schedule) | Run 10 (warm-start, no schedule) |
|------|------:|------:|------:|
| r_clearance | 0.113 | 0.025 | **0.124** |
| r_airtime | 0.006 | −0.023 | **0.007** |
| r_angvel | -0.067 | −0.165 | **−0.019** |
| r_yaw | -0.013 | −0.035 | **−0.003** |
| r_jointvel (raw) | 113 | 147 | **78** |
| r_abduct | -0.024 | −0.032 | **−0.013** |
| mean z | 0.310 | 0.284 | **0.317** |

**Problem analysis — the trot is a stable attractor; the schedule was a crutch, not a crutch-forever.**
Warm-started from the Run 8 trot with the gait reward off, the policy **keeps trotting** — `r_clearance`
0.124 and `r_airtime` positive sit at/above Run 8 levels, nothing like Run 9's shuffle (0.025 /
negative). More than persist, it **polished**: with no schedule reward to satisfy, the optimizer spent
that capacity minimizing the penalties, so buck (`r_angvel` −0.067→−0.019), yaw (−0.013→−0.003), joint
thrash (raw 113→78) and abduction all *improved* over Run 8, and height nudged up to 0.317. `std`
collapsed to **0.181** — an extremely committed, near-deterministic policy. **Conclusion:** the
diagonal-schedule reward was needed only to *reach* the trot from a random init (Run 9 proves it
can't be found cold); once the gait is learned it is self-sustaining and the reward can be removed.
This is the clean bookend to the Runs 6→9 story.

**Proposed solution → Run 11.** Flat walking is solved and robust. Move to **uneven terrain**:
warm-start from this flat policy and train on a randomized heightfield (curriculum). A blind
flat-trained policy will stumble at first — terrain support is added to `train_legacy.py` / `watch_legacy.py`
behind a `--terrain` flag (default off, so flat work is untouched).

---


<!-- TEMPLATE for future runs — copy the block below and fill in.

## Run N — <short name>

**Date:** · **Status:** · **Result:**

**Changed from previous:** (only what changed, and why)

**Problem analysis:** (key metrics / breakdown table; what the numbers + viewer show)

**Proposed solution → Run N+1:** (the single next change)

-->
