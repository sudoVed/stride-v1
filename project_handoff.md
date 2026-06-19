# STRIDE — Project Handoff

**STRIDE** is a reinforcement-learning project that teaches a **Unitree Go2** quadruped to walk in
**MuJoCo**, trained **from scratch with PPO**. Over a sequence of numbered *Runs* it grew from a robot
that couldn't stand into a single neural-network policy that trots, recovers from hard shoves, and
walks over uneven terrain. **The project is complete; what remains is packaging it for release** (§10).

---

## 0. The result (résumé / one-line summary)

> Trained a 15 kg Unitree Go2 quadruped to locomote from scratch with deep reinforcement learning
> (PPO in MuJoCo), developing a single neural locomotion policy for gait generation and disturbance
> recovery. The policy recovers from impulsive, unanticipated mid-gait pushes across 8 directions —
> 100% recovery at 90 N (~0.6× body weight) on flat ground, and 100% / 91.7% recovery at 75 N / 90 N
> on procedurally generated uneven terrain (0.15 m elevation).

What that means concretely: the policy is a single feed-forward network (no hand-coded controller). It
was trained by trial and error (PPO) starting from random behaviour — it learned to stand, then trot,
then stay up when shoved mid-stride, then traverse randomized bumpy ground. The shoves in the eval are
**impulsive and unannounced** (no command, no warning, applied while trotting), so recovery is purely
reactive — the hardest and most honest robustness test.

---

## 1. Status — COMPLETE (shipping)

The shipped model lives in **`python/final/`** (PPO weights + VecNormalize + reward breakdown). Final
eval (`eval_policy.py`, 24 episodes, 8 compass directions, 0.2 s torso shove mid-trot, 3 s window) on
the **15.21 kg** Go2 (149 N weight):

| impulse (8 directions, mid-gait) | flat ground | uneven terrain (0.15 m) |
|---|---|---|
| **75 N** (≈0.50× body weight) | — | **100 % (24/24)** |
| **90 N** (≈0.60× body weight) | **100 % (24/24)** | **91.7 % (22/24)** |

The only falls in the whole sweep are sideways / back-left (90°, 135°) shoves on terrain at 90 N —
lateral disturbance on uneven ground is the single hard case; the lateral weakness is expected for a
diagonal trot (narrow lateral support, legs cycle fore-aft). Full analysis: `report_logs_4.md`, **Final**
entry.

---

## 2. What the project is (and the journey)

Reinforcement learning for legged locomotion: the policy outputs joint targets at 50 Hz, the simulator
rolls physics forward, and PPO rewards forward progress + staying upright + a clean gait. Four phases,
each recorded in its own log:

1. **Flat walking, from scratch — `report_logs_1.md` (Runs 1–10).** Reward-hacked standing → faceplant
   → PD position control → crouch → shuffle → a prescribed diagonal-trot schedule → a clean converged
   trot. Lesson: each fix targeted the single quantity the reward left unconstrained.
2. **Disturbance robustness (fixed-clock) — `report_logs_2.md` (Runs 11–15).** Spawn-settle hold, push
   domain-randomization, and a **`d`-gated recovery mode** (an off-balance signal relaxes the tidy-gait
   penalties + lowers the height target so crouching/stepping-out become the *rewarded* response). Got
   to 100 % push recovery on flat.
3. **PMTG migration — `report_logs_3.md` (Runs 16–25).** Replaced the fixed gait clock with a
   **policy-modulated phase + foot-trajectory generator** (ANYmal/Lee 2020, Iscen 2018), so the gait is
   a *breakable prior the policy controls* — the prerequisite for adapting footfalls to terrain. New
   action/obs space (16-D / 60-D) → trained from scratch.
4. **Push recovery + terrain on the new architecture — `report_logs_4.md` (Runs 26 → Final).** Re-won
   push recovery (Run 26, 100 %), built the `d`-gated *structure* release for terrain (Run 27), then
   solved terrain with **box-tile terrain** after MuJoCo's heightfield collider proved unusable for a
   small foot. → the shipped `final/` model.

---

## 3. The final controller (architecture)

- **Control:** PD **position** control at 50 Hz (`frame_skip=10`, 0.002 s timestep, `KP=30, KD=0.75`).
  The policy outputs *offsets*, the PD law makes torque, so the robot holds itself up "for free."
- **PMTG / FTG:** one shared trot clock (`BASE_FREQ=1.5 Hz`) + fixed trot offsets `[0,0.5,0.5,0]` + a
  per-leg phase offset the policy drives (clamped to ±`DELTA_MAX`). A **Foot Trajectory Generator** adds
  the nominal swing arc; the policy adds **residuals** on top. Coordination + swing are *structural*
  (not emergent from reward) — the hard lesson of Runs 16–25.
- **Action (16-D):** 12 joint residuals + 4 per-leg frequency offsets. **Obs (60-D):** local-ground
  height, orientation, joint angles/velocities, base lin/ang vel, commanded speed, previous action, and
  the 4 per-leg phase `sin/cos`. VecNormalize on obs + reward.
- **`d`-gated recovery + released structure:** an off-balance signal `d` relaxes `r_pose`/`r_abduct`/
  `r_phase`/`r_track` and lowers the height target; under disturbance it **also fades the FTG swing arc
  and widens the phase clamp**, so a leg can break the fixed swing and retime to brace / step over a
  bump. Calm (`d=0`) leaves the clean trot untouched.
- **Foot:** a **0.035 m capsule** (the menagerie's 0.022 m sphere extended up the shank, radius bumped)
  so it lands on the box-tile tops and bridges inter-tile steps.
- **Terrain:** a forward strip of overlapping **box tiles as mocap bodies** (per-episode heights set via
  `data.mocap_pos`, so collision tracks them — a moving *static* geom would not), with **fractal
  multi-octave value-noise heights** and a flat spawn patch. (MuJoCo's heightfield collider was
  abandoned — unstable for a small foot; see discussions #2175 / #2307 and the Final entry.)
- **PPO:** MlpPolicy `[256,256]`, 8 envs, `n_steps=2048`, `batch=4096`, `n_epochs=10`, `γ=0.99`,
  `gae_λ=0.95`, `lr=3e-4`, `clip=0.2`, 3 M steps. Warm-starts raise `ent_coef→0.01` and reset action
  `std→0.5` for exploration.

---

## 4. Environment & setup

- **OS:** Windows (native, no WSL). **Python:** 3.12 (venv `py12venv`; 3.14 has no MuJoCo/SB3 wheels).
- **Key packages:** `mujoco` 3.9, `gymnasium` 1.3, `stable-baselines3` 2.8, `tensorboard`, PyTorch.
  Do **not** install the legacy `gym` package (breaks the install).

```
cd E:\STRIDE
py -3.12 -m venv py12venv
py12venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install mujoco gymnasium "stable-baselines3>=2.3.0" tensorboard
```

Always activate the venv before running anything.

---

## 5. File / folder layout

```
E:\STRIDE\
├── project_handoff.md          <- this file (entry point)
├── report_logs_1.md            <- Runs 1–10  : flat walking from scratch (+ base setup reference)
├── report_logs_2.md            <- Runs 11–15 : terrain (early) + fixed-clock push recovery
├── report_logs_3.md            <- Runs 16–25 : the PMTG / FTG migration
├── report_logs_4.md            <- Runs 26 → Final : Tier-3 push recovery + box-tile terrain (SHIPPED)
├── mujoco_menagerie\unitree_go2\
│   ├── go2.xml                 <- robot (CAPSULE foot 0.035, elliptic-cone contact)
│   ├── scene.xml               <- Go2 + flat floor (flat training/eval)
│   └── scene_terrain.xml       <- Go2 + BOX-TILE terrain (generated by make_box_terrain.py)
└── python\
    ├── train.py                <- MAIN: Tier-3 PMTG/FTG env + reward + PPO (--push, --terrain)
    ├── watch.py                <- viewer (--terrain supported)
    ├── eval_policy.py          <- quantitative push-recovery eval (--terrain, --terrain-height, --force)
    ├── make_box_terrain.py     <- generates scene_terrain.xml (box-tile forward strip)
    ├── terrain_preview.py      <- previews terrain difficulty (elevation / slope / steps) without training
    ├── final\                  <- ★ SHIPPED policy (3 M)  — the deliverable
    ├── checkpoints\            <- working dir (WIPED at the start of every run; rename to keep)
    └── legacy\
        ├── legacy_12D_output\  <- fixed-clock era: run8_gait, run14_push, run15_recovery
        │                          + train_legacy.py / watch_legacy.py / eval_policy_legacy.py (12-D scripts)
        └── legacy_16D_output\  <- Tier-3 milestones: run25_ftg (clean flat trot), run26_push_ftg,
                                   run27_push_ftg (d-gated push; the --terrain warm-start source) + run26_train.py
```

**Checkpoint hygiene:** `train.py` wipes `checkpoints\` at the start of every run — rename a run's
output to a descriptive folder before launching the next.

---

## 6. How to run

From `E:\STRIDE\python` with the venv active.

```
# Train (flat / push / terrain). Warm-start is set by WARM_START at the top of train.py.
python train.py
python train.py --push
python train.py --terrain --terrain-height 0.15

# Watch a checkpoint (arg 1-4 = 750k/1.5M/2.25M/3M, 0 = latest). --dir picks the folder.
python watch.py 4 --dir final
python watch.py 4 --dir final --terrain

# Quantitative push-recovery eval (writes eval_<N>_steps.txt, tagged [flat ground] / [terrain H]).
python eval_policy.py 4 --dir final                         # flat
python eval_policy.py 4 --dir final --terrain --terrain-height 0.15
python eval_policy.py 4 --dir final --terrain --force 0     # no-shove terrain-traversal survival

# Terrain tooling
python make_box_terrain.py        # regenerate scene_terrain.xml (edit XB/XF/YH/PITCH/HALF in the file)
python terrain_preview.py 0.15    # preview elevation / slope / step difficulty for a given height

# TensorBoard
tensorboard --logdir runs
```

**Terrain tuning lives in `train.py`:** `TERRAIN_XB/XF/YH/PITCH` (strip extent — must match
`make_box_terrain.py`) and `TERRAIN_OCTAVES` (fractal `(period_m, amplitude)` layers — smaller period =
faster-changing / rougher). Preview any change with `terrain_preview.py` before training.

---

## 7. The shipped eval (headline)

See §1 for the table. Saved as `python/final/eval_3000000_steps.txt` (flat) and the terrain variants,
each tagged with the ground condition. Protocol and failure-direction analysis: `report_logs_4.md`,
**Final** entry.

---

## 8. Research logs

Four logs, same per-run format (metadata → changed from previous → problem analysis with real numbers
→ proposed next change). `report_logs_1.md` opens with the full base setup/reference; the others record
deltas. The **Final** entry in `report_logs_4.md` is the capstone (results + résumé line + why terrain
was warm-started from the d-gated Run 27).

---

## 9. Reproducing / extending (notes for the next person)

- **Goal-directed locomotion (planned next):** the obs carries a *commanded forward speed*. Generalize
  it to a **commanded direction** (desired velocity vector or heading), feed the goal-relative direction,
  and the same policy steers toward a goal. Adding that input changes the obs shape, so don't retrain
  from scratch — do **weight surgery** (copy the existing weights into the larger network, zero-init the
  new input columns) and fine-tune; the locomotion transfers, only steering is learned.
- **Harder terrain:** raise `--terrain-height` and/or shorten `TERRAIN_OCTAVES` periods (curriculum);
  `terrain_preview.py` tells you if steps outgrow the foot, in which case enlarge the capsule.
- **Sandbox note:** the Linux helper sandbox sometimes serves a stale/truncated copy of `train.py`;
  trust the on-disk file and run a real `python -m py_compile` locally.

---

## 10. Remaining tasks (all that's left)

1. **Write the README** — the public-facing front page for the repo: what STRIDE is, the headline result
   (use the §0 résumé line + the §1 table), a short "how it works" (PPO from scratch → PMTG/FTG → push
   recovery → box-tile terrain), setup + run commands (§4, §6), and ideally a GIF/video of the policy
   trotting, recovering from a shove, and crossing terrain.
2. **Decide & prepare the files for upload** — choose what ships vs stays local:
   - **Include:** `python/` scripts (`train.py`, `watch.py`, `eval_policy.py`, `make_box_terrain.py`,
     `terrain_preview.py`, legacy scripts), the `unitree_go2` XMLs, the four `report_logs_*.md`, this
     handoff, the README, and the **`final/`** checkpoint (the shipped policy + its VecNormalize).
   - **Decide:** which other checkpoints to keep (e.g. `run25_ftg`, `run26_push_ftg`, `run27_push_ftg`
     as milestones) vs drop to keep the repo small; whether to ship `runs/` (TensorBoard logs).
   - **Exclude:** the venv (`py12venv/`), `checkpoints/` working dir, `__pycache__/`. Add a `.gitignore`.
   - Confirm `eval_<...>.txt` files for `final/` are present as evidence for the README claims.
```
