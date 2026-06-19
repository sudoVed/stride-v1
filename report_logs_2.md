# STRIDE — Quadruped RL Training Log (Uneven Terrain)

Phase 2 of the project: teaching the Unitree Go2 to walk on **uneven terrain**, continuing from the
solved flat-ground policy. The flat history, the base training setup, the full observation/action
spec, and the reward terms are in **`report_logs_1.md`** — this file records only what is
*different* for terrain plus the per-run terrain history.

**Starting point:** the **Run 8** flat trot (`run8_gait/ppo_go2_3000000_steps.zip`),
*not* Run 10 — Run 10 was over-converged (action `std` 0.181), leaving little room to re-learn;
Run 8 (`std` 0.352) has more exploration headroom to adapt to terrain.

**Same script/viewers:** `train_legacy.py` (`--terrain`), `watch_legacy.py` (`--terrain`).

---

# 1 · Terrain Setup (deltas from the flat policy)

Everything in `report_logs_1.md` §1 carries over (PPO hyperparameters, 50-D observation,
12-D action, PD control, the reward *set*) **except** the changes below.

## 1.1 Terrain & spawn

| Property | Value |
|---|---|
| Scene | `scene_terrain.xml` — flat floor replaced by an 80×80 heightfield (`--terrain`) |
| Max bump height | `--terrain-height`, default **0.06 m** (validated stable; 0.05 easy, 0.10 occasionally flips a *blind flat* policy) |
| Terrain shape | smooth random heightmap (box-blurred → walkable bumps, no knife-edges) |
| Spawn | a flat patch (~2 m) at the centre is forced to ground level, and the robot spawns 3 cm above it, so feet never start buried |
| Collision | foot↔heightfield handled by MuJoCo; foot geoms have contact priority |

## 1.2 Local-ground height (the key reward/obs change)

On flat ground, torso height = absolute world `z`. On terrain that's wrong — a torso standing
correctly on a bump reads "too high" and a dip reads "too low," so absolute `z` biases the height
reward and mis-fires the fall check. Fixed by measuring height **above the ground beneath the
robot**:

- A ray is cast **straight down from the robot COM**, with a `geomgroup` mask enabling **only group 0
  (the terrain floor)** — robot geoms are groups 2 (visual) / 3 (collision), so the ray hits the
  ground and never a leg. `ground_z = COM_z − ray_distance`, and `h_local = base_z − ground_z`.
- `h_local` replaces absolute `z` in **(a)** the height reward `0.5·exp(−(h_local−0.33)²/(2·0.05²))`,
  **(b)** the fall termination `h_local < 0.18`, and **(c)** the height **observation** input (so the
  policy senses true clearance). Dimension unchanged (still 50-D) → Run 8 loads fine.
- On flat ground `ground_z = 0`, so `h_local ≡ z` — the flat behaviour is byte-identical (no
  regression). Validated against MuJoCo: the ray correctly returns the ground geom and distance.
- The ray is exposed for `watch.py`, which draws it as a **red rod from the COM to the ground** so you
  can see what the policy is measuring.

## 1.3 Domain randomization (per episode)

Reset re-randomizes, so the policy meets variety and learns to recover rather than memorize:

| Knob | Range |
|---|---|
| Terrain heightmap | freshly regenerated every episode |
| Foot sliding friction | nominal (0.8) × `U(0.6, 1.4)` per foot → ~0.48–1.12 (slippery↔grippy) |

*(Deferred, per plan: sudden random push forces for explicit recovery training — add later.)*

## 1.4 Exploration (terrain warm-start)

The flat policy is over-converged, so on `--terrain` warm-start we re-open it:

- **`ent_coef` 0.0 → 0.01.** The *entropy coefficient* is the weight on an **entropy bonus** added to
  PPO's objective. Entropy measures how random the action distribution is; rewarding it discourages
  the policy from collapsing to one rigid behaviour and keeps it *trying variations*. 0 = "be as
  deterministic as the task allows" (fine once solved); a small positive value = "stay a bit
  exploratory," which is what we want when re-learning on new ground.
- **Action `std` reset to 0.5** (from the converged 0.181) so it *actually* attempts variations (e.g.
  recovery steps) from step one instead of clinging to the flat gait.

## 1.5 Reward terms (terrain)

Identical weights to flat Run 8, with two notes: **`r_height` now uses `h_local`** (§1.2), and
**`r_gait` is off (`W_GAIT = 0`)** — no prescribed gait on terrain for now; we want to see what it
does. All stability/effort penalties (`r_orient` −0.5, `r_angvel` −0.05, `r_yaw` −0.15,
`r_jointvel` −1.5e-3, `r_abduct` −0.5, `r_pose` −0.02, `r_actrate` −3e-4, `r_torque` −1e-5) and the
positives (`r_vel` 1.5, `r_height` 0.5, `r_alive` 0.1, `r_airtime` 0.5, `r_clearance` 0.10) are
unchanged from flat.

## 1.6 How to run

```
python train_legacy.py --terrain                    # warm-starts Run 8, 0.06 m bumps
python train_legacy.py --terrain --terrain-height 0.05
python watch_legacy.py 4 --terrain                   # replay with the red ground-ray visible
```

---

# 2 · Run History

---

## Run 11 (terrain) — flat trot → randomized bumps (local-ground height + DR)

**Date:** 2026-06-13 · **Warm-start:** flat Run 8 (3M) · **Status:** ran to 3M · **Result:** learns terrain well by every metric, but **visually messy — feet/limbs clip through the heightfield at high terrain settings.** (Collision artifact, not a learning failure.)

**Changed from flat Run 8:**
- Heightfield terrain (`--terrain`, 0.06 m), flat-patch spawn.
- Height reward / termination / observation now use **local-ground height** (COM down-ray, §1.2).
- **Domain randomization:** fresh terrain + foot friction every episode (§1.3).
- **Exploration boost:** `ent_coef` 0.01, action `std` reset to 0.5 (§1.4).
- `W_GAIT = 0` (no gait reward); all other rewards/penalties unchanged from Run 8.

**Reward/penalty breakdown @ 3.0M** (`r_height` is now height above LOCAL ground):

| term | weight | mean/step | share |
|------|-------:|----------:|------:|
| r_vel        |  1.5    |  1.4891 | 72.6% |
| r_height     |  0.5    |  0.4816 | 23.5% |
| r_alive      |  0.1    |  0.1000 |  4.9% |
| r_airtime    |  0.5    |  0.0068 |  0.3% |
| r_clearance  |  0.10   |  0.1268 |  6.2% |
| r_gait       |  0.0    |  0.0000 |  0.0% |
| r_orient     | −0.5    | −0.0008 | −0.0% |
| r_angvel     | −0.05   | −0.0118 | −0.6% |
| r_yaw        | −0.15   | −0.0025 | −0.1% |
| r_action_rate| −3e-4   | −0.0004 | −0.0% |
| r_torque     | −1e-5   | −0.0080 | −0.4% |
| r_jointvel   | −1.5e-3 | −0.1070 | −5.2% |
| r_pose       | −0.02   | −0.0102 | −0.5% |
| r_abduct     | −0.5    | −0.0114 | −0.6% |
| **TOTAL**    |         | **2.0522** | |

mean vx 0.737 m/s · mean local height 0.318 m.

**Problem analysis — the learning worked; the contact does not (at high amplitude).**
On the numbers this is a success: velocity tracks perfectly (`r_vel` 1.489 ⇒ vel_track ~0.99, vx 0.737),
the **local-ground height fix works** (mean `h_local` 0.318 m, right at target, with `r_height` 0.482 —
on terrain the *absolute* height would have been all over the place, so this confirms §1.2), it's very
stable (`r_angvel` −0.012, `r_yaw` −0.003 — tiny), feet lift (`r_clearance` 0.127, positive), and legs
stay under the body (`r_abduct` −0.011). The warm-started trot genuinely adapted to randomized bumps +
friction. `r_jointvel` raw ~71 — calm joints.

**But the geometry is wrong at high `--terrain-height`:** the feet and lower legs **clip through /
penetrate the heightfield**. This is a MuJoCo contact issue, not a policy issue — a small foot sphere
vs. a heightfield made of triangulated prisms allows penetration on steep faces, made worse by soft
contacts, fast swing speed, and the per-episode terrain re-randomization (whose render can also lag the
physics). It ruins the look for the comparison video even though the gait is good.

**Proposed solution → Run 12.** Eliminate terrain clipping (approach under decision — see candidates
below). Likely some combination of: gentler/finer heightfield (more smoothing, higher resolution),
stiffer/earlier contacts (contact `margin`, `solref`/`solimp`), smaller physics timestep, better foot
collision geometry, and ensuring the viewer re-uploads the hfield each episode.

---

## Run 12 (terrain) — eliminate terrain clipping via contact margin

**Date:** 2026-06-13 · **Warm-start:** Run 11 (terrain) · **Status:** implemented / running · **Result:** (pending).

**Goal:** keep Run 11's good terrain gait, fix the foot/limb penetration ("clips into the ground and
gets stuck" at high `--terrain-height`) so contacts are accurate and the motion looks clean.

Run 12a applied a contact margin; the margin alone proved insufficient at high amplitude (jittery
contacts, broken spawn at 0.2 m, occasional clip-through at 0.1 m). Run 12b traced the real cause —
the *terrain was too steep* — and bounded its slope.

**Changed from Run 11:**
- **(12a) Contact margin on the ground geom: `TERRAIN_MARGIN = 0.015 m`** (`model.geom_margin[floor]`,
  max-of-pair so it covers feet + lower legs). Repels geoms just before they reach the surface.
  Necessary but not sufficient on its own.
- **(12b) Gentler, low-frequency terrain.** `_make_heightfield` now builds the map from a small
  `coarse=10` random grid (upsampled + 6 blur passes) instead of per-cell random noise. A per-cell
  random field at 0.2 m has near-vertical faces (≈0.2 m rise across a 0.15 m cell); the low-frequency
  version makes wide, walkable hills. **Measured max slope dropped 32° → 15°** at 0.2 m — those
  steep outlier faces were what the small foot clipped/jittered on.
- **(12b) Feathered flat spawn patch.** The old hard-edged flat patch was a *cliff* up to
  `terrain_height` tall at its border (→ broken spawn at 0.2 m). Replaced with a radial mask: flat at
  the centre, ramping smoothly to terrain over a feather band — no cliff.
- **(12b) Spawn lift 0.03 → 0.06 m** so feet start clear of the contact margin + reset noise (no
  spawn-time repel jitter).

**Validation (MuJoCo, headless):** at the worst setting (`--terrain-height 0.2`) the robot now spawns
with **0 penetration / 0 contacts** and settles **perfectly level** (torso up-vector 0.999) without
the solver exploding. Max terrain slope 15°, spawn patch perfectly flat.

**Trade-offs / remaining levers (if needed):**
- A swing foot may still hover up to ~`TERRAIN_MARGIN` (1.5 cm) above the surface; lower it if floaty.
- If jitter/clip persists at extreme amplitude, the untried levers are: stiffer/softer `solref`/`solimp`
  on the foot contact, a capsule foot geom (bridges small features), a smaller physics timestep, and a
  finer hfield resolution (XML `nrow/ncol`) for a better collision mesh.

**Problem analysis → (pending — watch at 0.1 and 0.2 to confirm clip/jitter is gone).**
**Proposed solution → (pending).**

---


## Run 13 — Spawn-settle hold + push-disturbance training

**Date:** 2026-06-13 · **Status:** FAIL · **Result:** absorbs weak/medium shoves, but strong ones flip it onto its back with no recovery — it never lowers its COM or steps a foot under the torso.

**Changed from previous (the Run 10 flat trot):** added the first two robustness features on flat ground, bundled into one run because spawn immunity alone is too small to train on its own. (1) **Spawn-settle hold:** each episode spawns ~5 cm high and the env commands the default pose for 0.5 s while termination + movement rewards are suspended, so the robot drops and settles before the policy must act. (2) **Push-disturbance DR (`--push`):** random world-frame torso shoves, 30–120 N for 0.2 s every 2–4 s, never during the hold; reward unchanged. (3) **Warm-start exploration boost:** `ent_coef=0.01`, action `std` reset to 0.5, so the over-converged trot has room to learn recovery. (Terrain Runs 11–12 are the separate branch; this resumes the flat-robustness line.)

**Problem analysis — recovers small, dies on large; reward forbids the recovery posture.** At 557k steps: `ep_len_mean` **557** vs the 1000-step cap — episodes are routinely cut short by falls, unlike Run 10 which ran to the cap. `ep_rew_mean` 841. `std` **0.576** (the boost held; exploration is healthy, not collapsed), `approx_kl` 0.0199 and `clip_fraction` 0.237 (high-ish — the policy is being pushed around a lot, consistent with learning under disturbance), `explained_variance` 0.775. Viewer: weak shoves are absorbed with small ankle/hip corrections; strong shoves tip it past the `up < 0.4` termination in one step and it lands on its back and stays there. **Root cause is the reward, not training:** `r_abduct` (−0.5·hip-abduction) penalizes throwing a leg out wide to catch a sideways push, `r_height` rewards staying at 0.33 (penalizes dropping the COM), and `r_pose` pulls joints back to the nominal stance — i.e. the three terms that produced the tidy trot actively forbid the wide-stance, low-COM, foot-repositioning behaviours that recover from a big shove. The policy has no licensed recovery move, so beyond the small-correction envelope it just falls. (Per-term breakdown not captured this run — Ctrl+C now writes a `reward_breakdown_0_steps.txt` so the next run has the term-level numbers.)

**Proposed solution → Run 14:** a **disturbance-gated recovery mode**. An off-balance signal `d ∈ [0,1]` (from torso tilt + roll/pitch rate, with deadzones so `d=0` in calm walking) **relaxes** `r_pose` and `r_abduct` by `(1−d)` and **lowers the height target** by up to 0.10 m·`d`, so crouching and wide foot placement become the *rewarded* response when shoved while the clean trot is untouched when calm. Also cap/curriculum `PUSH_FORCE_MAX` to the recoverable envelope (use `eval_policy.py --force` sweep to find the cliff; ~120 N ≈ 1.6 m/s lateral is likely past the physical limit). If COM-lowering + penalty-relaxation isn't enough, add a support-polygon term (reward COM ground-projection over the stance-foot centroid, gated by `d`) to induce the stepping strategy.

---

## Run 14 — Disturbance-gated recovery mode (relax penalties + lower COM)

**Date:** 2026-06-13 · **Warm-start:** Run 8 flat trot (`run8_gait`), `--push` + exploration boost · **Status:** SUCCESS · **Result:** recovers ~92 % of 90 N side shoves by dropping its COM and stepping under itself; clean and self-righting. One residual cosmetic fault: a small lone-FL pre-hop in the calm trot (reads as a slight limp).

**Changed from previous (Run 13):** added the **disturbance-gated recovery mode** that Run 13 prescribed. A proprioceptive off-balance signal `d ∈ [0,1]` is computed each step from torso tilt (`1−up`) and roll/pitch rate, with deadzones (`D_TILT_DEAD 0.05`, `D_RATE_DEAD 1.5`) so `d = 0` during calm walking and the clean trot is untouched. When `d` rises: `r_pose` and `r_abduct` are scaled by `(1−d)` (freeing the legs to splay/step out to catch a shove), and the height target is lowered by `HEIGHT_DROP_MAX·d` (0.33 → 0.23 m fully off-balance), making a COM drop the *rewarded* response. Spawn-settle hold + push DR (30–120 N) unchanged. Also: Ctrl+C now writes a `reward_breakdown_0_steps.txt`.

**Key metrics (3.0M):** `ep_rew_mean` **1200**, `ep_len_mean` **846**, `std` 0.978, `entropy_loss` −16.6, `approx_kl` 0.0144, `clip_fraction` 0.182, `expl_var` 0.773. The reward is *lower* than Run 8's calm-walking 2330 — expected, **not a regression**: (1) the task is now much harder — every episode is repeatedly shoved, and during/after each shove velocity falls off target, the COM is deliberately dropped (so `r_height` sits below its 0.33 peak) and `r_angvel` buck rises (−0.196 vs Run 8's −0.019), all of which cost per-step reward; (2) `ep_len_mean` 846 < 1000, so episodes that end in a fall accumulate less total reward; (3) `std` is deliberately high (0.978, from the exploration boost), so the policy is still acting stochastically. Reward is only comparable *within the same task* — the meaningful read is Run 13 → 14: `ep_rew` 841 → 1200 and `ep_len` 557 → 846 (far fewer falls), plus the 92 % eval recovery.

**Problem analysis — recovery solved; only a gait-symmetry artifact remains.** Eval (`eval_policy.py`, 24 episodes, scripted 90 N / 0.2 s side shove): **fall rate 8.3 % (2/24), recovery rate 91.7 % (22/24), 0 % stayed-up-unsteady, mean dip 0.292 m**. The 0.292 m dip vs the 0.33 m target is direct evidence the policy *is* lowering its COM under load, exactly as the `d`-gate intended — Run 13's flip-and-die is gone. Breakdown @ 3M: `r_vel` 1.361 (97 %), `r_height` 0.420, `r_clearance` 0.099 (real stepping intact), `r_angvel` −0.196 (some residual buck, ~14 %), `r_gait` **0.000**. That last zero is the cause of the FL hop: with the diagonal-trot schedule reward off (`W_GAIT=0`), nothing constrains the two feet of a diagonal pair to move together, so FL is free to insert a solo pre-hop (lifting while RR is still planted). It is *not* a reward exploit — `r_airtime ≈ 0` and the hop earns nothing — just an un-penalised asymmetry the optimizer never had a reason to remove.

**Proposed solution → Run 15:** re-introduce a **small diagonal-trot schedule reward, gated by `(1−d)`** — `W_GAIT 0 → 0.15`, and multiply `gait_reward` by `relax` so it only shapes the calm gait and fades during recovery (un-gated it would re-freeze the recovery footwork and undo the 92 %). Rewarding FL+RR synchronization makes the lone-FL pre-hop unprofitable, training out the limp. Warm-start from **this Run 14** policy (`run14_push`) with `--push` still on, so the recovery is preserved and only the gait is polished. If the hop persists, raise `W_GAIT` toward 0.3; if the fixed 1.5 Hz cadence feels forced, switch to a phase-free diagonal-pair contact-sync penalty instead.

**★ Preserved checkpoint (key POI):** `python/legacy/legacy_12D_output/run14_push/` (3M; was `checkpoints-push`). The d-gated recovery policy — 92 % push recovery; warm-start source for Run 15. Worth keeping as the "recovery learned" milestone.

---

## Run 15 — Gated diagonal-gait reward (fix the FL pre-hop)

**Date:** 2026-06-14 · **Warm-start:** Run 14 (`legacy_push`, 3M), `--push`, trained **1.5M** (polish only, warm-started so no full 3M needed) · **Status:** SUCCESS · **Result:** the lone-FL pre-hop is gone — the trot reads clean and symmetric — and recovery was *preserved and improved*, not lost.

**Changed from previous (Run 14):** re-enabled the diagonal-trot schedule reward — `W_GAIT 0 → 0.15` — but **gated by `(1−d)`** (`gait_reward` multiplied by `relax`), so it only shapes the calm gait and fades during recovery (un-gated it would re-freeze the recovery footwork). Single conceptual change; everything else (spawn hold, push DR, `d`-mode) unchanged.

**Key metrics (1.5M):** `ep_rew_mean` **1580**, `ep_len_mean` **939** (near the 1000 cap — few falls), `std` 0.709, `approx_kl` 0.0164, `clip_fraction` 0.207, `expl_var` 0.815. Breakdown @1.5M: `r_gait` now **+0.0997** (5.9 % — the schedule is actively being followed, vs 0.000 in Run 14, i.e. the desync hop is being trained out), `r_angvel` improved −0.196 → **−0.138** (synchronized pairs cut the roll/pitch buck), `r_jointvel` −0.232 (now the largest single penalty), mean vx 0.697, mean z 0.308.

**Problem analysis — hop fixed, recovery untouched (improved).** The gated gait reward did exactly its job: `r_gait` going positive means the policy now satisfies the diagonal schedule, removing the lone-FL pre-hop, and it cost nothing elsewhere (orientation, pose, abduction all near-zero). Eval (latest save, `eval_policy.py`, 90 N side shoves): **fall rate 0 % (0/24), recovery rate 100 % (24/24), mean dip 0.292 m** — up from Run 14's 91.7 %. The `(1−d)` gate is confirmed working: imposing rhythm when calm did *not* degrade recovery (the gait term releases under disturbance), and the extra 1.5M of training even nudged recovery to 100 %. `ep_len` 846 → 939 and `ep_rew` 1200 → 1580 both rose — fewer falls + cleaner motion.

**★ Preserved checkpoint (key POI):** `python/legacy/legacy_12D_output/run15_recovery/` (1.5M, the current run's output). **Best flat policy to date** — clean symmetric trot *and* 100 % push recovery. The reference "flat robustness solved" milestone.

**Proposed solution → next (major direction change):** flat robustness is solved (walk + recover + clean gait). The next big step is the **ANYmal PMTG migration** — replace the fixed gait clock with a *policy-controlled* per-leg phase (up to Tier 2) so the gait can retime itself, which is the prerequisite for a robust return to terrain. This changes the action/observation space, so it breaks warm-start weight compatibility (from-scratch train, or weight surgery) — a deliberate, larger effort tracked separately. See `report_logs_3.md` for the PMTG migration plan.

---

<!-- TEMPLATE for terrain runs — copy below and fill in.

## Terrain Run N (TN) — <short name>

**Date:** · **Warm-start:** · **Status:** · **Result:**

**Changed from previous:**

**Problem analysis:** (metrics / breakdown; viewer behaviour — recoveries, flips, footfall)

**Proposed solution → TN+1:**

-->
