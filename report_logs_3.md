# STRIDE — Quadruped RL Training Log (Policy-Modulated Phase / PMTG)

Phase 3 of the project: replacing the **fixed gait clock** with a **policy-controlled per-leg phase**
(ETH ANYmal-style PMTG, Tier 2), so the gait can *retime itself*. This is the prerequisite for a
robust return to uneven terrain — a fixed clock cannot adapt footfall timing to bumps, but a
self-timed one can.

The base training setup (PD position control, 50 Hz, PPO hyperparameters, the reward *set*, the
spawn-settle hold, the push DR, the `d`-gated recovery mode) is unchanged from the flat-robustness
line and is documented in **`report_logs_1.md`** (flat, Runs 1–10) and **`report_logs_2.md`**
(terrain 11–12 + flat robustness 13–15). This file records only what is *different* for the PMTG
migration plus the per-run history from **Run 16** onward.

**Starting point:** trained **from scratch** — Tier 2 changes the action space (12 → 16) and the
observation space (50 → 60), so the `run15_recovery/` (Run 15) weights **cannot** load (wrong
input/output layer shapes). The fixed-clock scripts are preserved as `train_legacy.py` /
`watch_legacy.py` / `eval_policy_legacy.py`; the new action-space versions are `train.py` /
`watch.py` / `eval_policy.py`.

---

# 1 · PMTG Setup (deltas from the flat policy)

Everything in `report_logs_1.md` §1 carries over **except** the action space, the observation
space, and the gait reward.

## 1.1 The idea — a *breakable rhythmic prior*, not a scripted clock

**Legacy (Runs 6–15).** One global phase `self._phase` advanced at a **fixed** `GAIT_FREQ = 1.5 Hz`.
The policy could only *read* its `sin/cos`, and a diagonal-trot reward `r_gait` **punished** any
departure from a hardcoded schedule (FL+RR down for the first half-cycle, FR+RL for the second). The
rhythm was external and un-retimable: the dog could not choose to stall one leg to step over an
obstacle, because the clock ran on rails and the reward enforced the rails.

**Tier 2 (this phase).** Each leg gets its **own** phase, and the phase becomes a **state variable
the policy controls**. The policy now outputs four extra numbers that set each leg's phase-advance
*rate*. The gait is therefore a *prior the policy can bend or break*: on flat ground it settles into
a steady trot-like rhythm; on rough ground (or mid-recovery) it can speed up, slow down, or freeze
an individual leg's clock to place that foot where it's needed. This is exactly the mechanism behind
the ANYmal papers' emergent **foot-trapping reflex** (snag a foot mid-swing → retract and re-place):
it *emerges* from giving the policy phase authority, rather than being scripted.

## 1.2 What references it, and why it works

- **Hwangbo et al. 2019, "Learning agile and dynamic motor skills for legged robots"** (ANYmal) and
  **Lee et al. 2020, "Learning quadrupedal locomotion over challenging terrain"** (ANYmal-C). Both
  drive the legs with a **phase / trajectory generator** whose timing the policy modulates, instead
  of tracking a fixed reference gait. The rhythm gives the optimizer a strong, sample-efficient
  *prior* (don't rediscover "legs move periodically" from scratch), while leaving the policy free to
  deviate where the terrain demands it. The project's PMTG migration plan and the parked-backlog note
  both point at this.
- **Iscen et al. 2018, "Policies Modulating Trajectory Generators" (PMTG)** — the framing this is
  named after: a trajectory generator emits an open-loop periodic motion and the policy modulates its
  parameters (here, the per-leg frequency) and adds residuals (here, the PD offsets). Tier 2 is the
  "modulate the generator's *frequency*, per leg" slice of that idea; Tier 3 (full TG + residuals)
  is deferred.

**Why it works (the mechanism, concretely).** A periodic phase is the right inductive bias for
locomotion because gait *is* periodic — handing the policy a clock removes the hard exploration
problem of inventing rhythm from a flat reward. But a *fixed* clock over-constrains: it bakes in one
cadence and one inter-leg pattern, which is fine on flat ground and wrong on bumps. Tier 2 keeps the
prior (there is still a clock per leg, initialized to a trot offset) but moves the *control* of the
clock inside the policy, so the same network that decides joint targets also decides timing. Because
timing and motion are now co-optimized against the same reward, the policy can trade them off —
e.g. stall a leg's phase (freeze that foot in stance) to keep three feet down while the body rides
out a disturbance, then resume. That trade-off is *unrepresentable* with a fixed clock.

## 1.3 Architecture (the concrete changes)

**Action: 12 → 16.**

| dims | meaning | maps to |
|---|---|---|
| 0–11 | normalized joint offsets in [−1,1] (unchanged) | `q_target = default_pose + ACTION_SCALE·a[:12]`, then the PD law |
| 12–15 | per-leg **frequency modulation** in [−1,1], order FL, FR, RL, RR | `f_i = GAIT_FREQ + FREQ_RANGE·a[12+i]`, clamped to `[0, FREQ_MAX]` |

With `GAIT_FREQ = 1.5`, `FREQ_RANGE = 1.5`, `FREQ_MAX = 3.5`: `a_i = −1` ⇒ `f_i = 0` (the leg's
clock **freezes** — it can stall a phase), `a_i = 0` ⇒ nominal 1.5 Hz, `a_i = +1` ⇒ 3.0 Hz (capped
at 3.5). Each leg's phase then integrates `φ_i ← (φ_i + f_i·dt) mod 1`.

**Observation: 50 → 60.** The single phase `sin/cos` (2) becomes **four** phases `sin/cos` (8, laid
out `[sin×4, cos×4]`), and `prev_action` grows 12 → 16. Full layout:
`height(1) + quat(4) + joints(12) + lin_vel(3) + ang_vel(3) + joint_vel(12) + v_target(1) +
prev_action(16) + per-leg phase sin/cos(8) = 60`.

**Phase init (a gentle prior, not enforced).** Each episode initializes the four phases to the
trot-like offsets `[FL, FR, RL, RR] = [0.0, 0.5, 0.5, 0.0]` plus a global random rotation and small
per-leg jitter — a sensible diagonal start the policy is free to leave. During the spawn-settle hold
all four advance at the nominal `GAIT_FREQ` (the policy isn't driving yet).

## 1.4 How the phase manages itself — and the role of reward

This is the crux, so it's worth stating precisely.

**The clock is self-managed by the policy.** Each `φ_i` is a small integrator the env carries; the
policy *writes its rate* (action 12–15) and *reads its value* (`sin/cos` in the obs). Nothing external
moves it on rails anymore. That alone, though, would make the phase *meaningless* — if no reward ever
referenced it, the four phase inputs would be decorative and the four frequency outputs would have no
gradient, so the policy would ignore them and we'd be back to a memoryless controller with four dead
action dims.

**So we replace the prescriptive gait reward with a weak *self-consistency* reward, not nothing.**
The old `r_gait` (fixed FL+RR / FR+RL diagonal schedule, ±penalty) is **removed**. In its place,
`r_phase` rewards each leg for being in **stance during the stance half of *its own* phase** (`φ_i <
0.5`) and **swinging during the swing half** (`φ_i ≥ 0.5`):

```
expected_stance_i = (phi_i < 0.5)                       # per leg, from the policy's OWN clock
phase_match       = mean_i( contact_i == expected_stance_i )   # in [0,1]
r_phase           = (1 - d) · W_PHASE · (2·phase_match - 1)     # centered to [-1,1], d-gated
```

The critical difference from the legacy term: **it couples each leg only to its own clock, never to
the other legs.** It says "make your declared phase honest" (be down when your clock says stance),
not "trot like this." Consequences:

- The **inter-leg pattern is emergent.** Trot, the timing offsets, per-leg retiming on terrain — none
  of it is dictated. The policy can converge to a diagonal trot (and likely will, from the init and
  because it's efficient) or depart from it when the terrain pays for departing.
- The **frequency action becomes learnable.** Because contact is now coupled to phase, "speed up this
  leg's clock" literally means "lift this foot sooner," and `r_phase` gives gradient on getting that
  right. The velocity / air-time / clearance terms supply the pressure to actually step; `r_phase`
  supplies the rhythm and makes the timing channel functional.
- It's **`d`-gated** (`×(1−d)`) exactly like the legacy gait/pose/abduction terms: under a real
  disturbance the rhythm prior *releases*, so recovery footwork is never fought by the clock — the
  same gate that gave Run 14/15 their 100 % recovery.

So the honest answer to "no rewards needed for gait?": we drop the **prescriptive** gait reward, but
we keep one **weak, non-prescriptive** coupling (`r_phase`) so the self-controlled clock stays
anchored to reality and the timing action has a gradient. Set `W_PHASE = 0` to test the pure-emergent
extreme (clock as pure observation prior); expect a slower, messier convergence and a real risk the
policy ignores the clock entirely. A tiny `r_freq = −(1−d)·W_FREQ·Σa_freq²` regularizer also keeps
the cadence from thrashing early on, and releases under disturbance so per-leg retiming stays free.

## 1.5 Reward table (deltas only)

Unchanged from Run 15: `r_vel`, `r_height` (with `d`-lowered target), `r_alive`, `r_airtime`,
`r_clearance`, `r_orient`, `r_angvel`, `r_yaw`, `r_action_rate` (now over all 16 dims, so it also
smooths the frequency commands), `r_torque`, `r_jointvel`, `r_pose` (×`(1−d)`), `r_abduct` (×`(1−d)`).

| Term | Weight·form | Change |
|---|---|---|
| `r_phase` | `(1−d)·W_PHASE·(2·match−1)`, `W_PHASE = 0.15`, **per-leg self-consistency** | **replaces** `r_gait` (fixed diagonal schedule) |
| `r_freq` | `−(1−d)·W_FREQ·Σa_freq²`, `W_FREQ = 0.01` | **new** — keeps the clock steady when calm; released under disturbance |

## 1.6 Same scripts / how to run

```
python train.py                 # Tier-2, FLAT, from scratch (WARM_START = None), 3M steps
python train.py --push          # add push-disturbance DR + d-gated recovery
python train.py --terrain --terrain-height 0.06   # uneven terrain (after a flat Tier-2 walk exists)
python watch.py 4 --dir checkpoints               # watch the latest checkpoint (16-D handled automatically)
python eval_policy.py 4 --dir checkpoints          # quantitative push-recovery eval
```

Checkpoint hygiene is unchanged: `train.py` wipes `checkpoints/` at the start of every run — rename a
run's output to a `legacy_*` folder before launching the next.

## 1.7 Tier-3 update (Run 18+): the FTG architecture, and why we left pure Tier 2

Runs 16–17 ran the Tier-2 design above (independent per-leg phases; coordination + swing arc expected
to *emerge* from rewards). It failed the way the ANYmal literature predicts — strut (Run 16), then an
uncoordinated high-step once the feet lifted (Run 17). Reading **Lee et al. 2020** (ANYmal-C) settled
the question of how that line keeps a clean, coordinated gait: **it doesn't let those things emerge —
it makes them structural.** Lee 2020 defines a periodic phase per leg, `φ_i = φ_{i,0} + (f_0 + f_i)·t`,
with **one shared base frequency `f_0 = 1.25 Hz`**, the policy outputting only a bounded per-leg
frequency offset `f_i`; the **trot coordination is the fixed initial offsets `φ_{i,0}`**; and a **Foot
Trajectory Generator** `F(φ)` emits the nominal foot motion ("the FTG drives the vertical stepping
motion"), with the policy outputting **residuals** on top (their 16-D action = 4 leg frequencies + 12
foot-position residuals). This is the PMTG architecture of **Iscen et al. 2018**.

From Run 18 we adopt that structure, adapted to our joint-space PD (no IK):

- **Shared base clock + fixed trot offsets.** `BASE_FREQ = 1.5 Hz` for all legs; phases initialize to
  `TROT_OFFSET = [0, 0.5, 0.5, 0]` (FL+RR vs FR+RL) plus one shared random rotation, so the diagonal
  *relationship* is preserved every episode. The policy adds a bounded per-leg frequency offset
  (`f_i = BASE_FREQ + FREQ_RANGE·a`, clamped `[0, FREQ_MAX]`; `a = −1` stalls a leg). Coordination is
  now **structural** — the legs can't free-run apart, fixing Run 17's stumble at the root.
- **Joint-space FTG.** During the swing half of a leg's phase the FTG adds a nominal joint offset
  (thigh lift + knee flexion) that raises the foot on a `sin` arc, zero during stance:
  `q_target = default + FTG(φ) + RESIDUAL_SCALE·a[:12]`. The foot steps **by construction**, fixing
  the strut. Amplitudes (`FTG_THIGH_AMP 0.29`, `FTG_CALF_AMP −0.55`) were tuned by forward kinematics
  to a nominal **~0.10 m** peak foot lift; `RESIDUAL_SCALE = 0.5` lets the policy refine it.
- **Action / obs unchanged in shape:** still 16-D (now 12 joint *residuals* + 4 frequency offsets) and
  60-D, so `watch.py` / `eval_policy.py` need no change. `r_phase`/`r_clearance` drop to **auxiliary**
  weights (`W_PHASE 0.15`, `W_CLEARANCE 0.10`) since the FTG now supplies arc + coordination; a small
  `W_FREQ = 0.02` keeps the legs locked to the shared clock when calm and releases under disturbance.

## 1.8 Design note — why a trajectory-generator prior, not end-to-end

A fair objection to the FTG is that it looks hardcoded: a fixed phase clock and a fixed nominal leg
arc, with the policy only deviating from it. Two reasons this is the right call here, and one
clarification of how little is actually "hard".

**1. End-to-end is exactly what Runs 16–17 were, and it failed at our compute budget.** A blank-slate
policy that outputs raw joint targets with no periodic prior *can* learn good gaits — Rudin et al.
2021 ("Learning to walk in minutes", ETH), Margolis/MIT, and the parkour works all do it — but they
lean on **massively parallel simulation**: thousands of environments and on the order of billions of
steps in GPU sim (Isaac Gym), plus curricula and heavy domain randomization, to brute-force their way
out of the bad local optima (the strut, the uncoordinated stumble) that a flat reward creates. This
project runs **8 environments / 3M steps in MuJoCo on a single machine** — three to four orders of
magnitude less experience. At that budget the blank-slate search gets stuck, which is precisely what
Runs 16–17 showed. The FTG is a **sample-efficiency prior**: it injects the one thing we already know
for certain about locomotion — that it is periodic and diagonally coordinated — so PPO spends its
limited samples learning balance, foot placement, push recovery and terrain adaptation instead of
re-discovering "legs move up and down in turn" from scratch.

**2. The FTG is a baseline the policy overshadows, not a scripted controller.** The nominal arc is
just the *starting point* of each step; the learned residual authority is `RESIDUAL_SCALE = 0.5 rad`
against an FTG arc of only `0.29 / 0.55 rad`, so the policy's output is **as large as or larger than
the nominal** — it can amplify the step, flatten it, move the foot, or cancel the FTG outright with an
opposite residual. The per-leg frequency channel is genuine learned control too (stall or speed up a
leg's clock → the retiming / foot-trapping behaviour on terrain). In practice the FTG dominates only
in the earliest training (it's what stops the strut), and the policy's weights quickly grow to
overshadow the nominal commands. So the gait is *biased*, not *scripted* — the FTG sets the basin,
the network does the locomotion.

**3. It's faithful to the paper we're following, and the field does both.** Lee 2020 (ANYmal-C, the
terrain paper) uses exactly this PMTG/FTG structure; ETH's own massively-parallel work (Rudin 2021)
drops the TG and goes end-to-end. The split is about compute, not correctness. Given our single-
machine MuJoCo budget, the TG prior is the pragmatic, literature-backed choice; if compute ever
allowed thousands of parallel envs, a blank-slate run + curriculum would be the purer experiment.

---

## Run 16 — Tier-2 PMTG: policy-modulated per-leg phase (from scratch)

**Date:** 2026-06-14 · **Warm-start:** none (action/obs space changed — trained from scratch) ·
**Status:** PARTIAL · **Result:** learns a **stable forward shuffle** — survives full episodes and
tracks commanded speed (~0.66 m/s) — but it **struts**: feet skim the ground with no real swing arc,
and the per-leg phase clock is effectively **ignored**. Walks, but not the gait we want.

**Changed from previous (Run 15, the fixed-clock trot + 100 % recovery):** the gait clock is no
longer external. (1) **Action 12 → 16:** four per-leg frequency-modulation dims; each leg's phase
advances at `f_i = GAIT_FREQ + FREQ_RANGE·a_i ∈ [0, FREQ_MAX]` (`a_i = −1` freezes a leg, `+1` runs
it at 2× nominal). (2) **Obs 50 → 60:** four phases' `sin/cos` (8) replace the single phase's (2);
`prev_action` 12 → 16. (3) **Reward:** the fixed-diagonal `r_gait` is **removed** and replaced by a
non-prescriptive per-leg phase-consistency reward `r_phase` (`W_PHASE = 0.15`, `d`-gated), plus a
small clock-thrash regularizer `r_freq` (`W_FREQ = 0.01`, `d`-gated). Everything else (PD control,
spawn-settle hold, push DR, `d`-gated recovery, all other reward terms) is unchanged. This is a
deliberately bundled migration run — the action/obs/reward changes are inseparable — so it breaks
the usual one-change-per-run rule on purpose, and breaks warm-start (hence from scratch).

**Key metrics (1.5M):** `ep_len_mean` **1000** (no falls — fully stable), `ep_rew_mean` **1400**,
`std` 0.706, `approx_kl` 0.0157, `clip_fraction` 0.212, `expl_var` **0.951** (the value function is
already confident in this gait — it has settled into an attractor, not still searching),
`entropy_loss` −17. Training was clean; the problem is *which* gait it converged to.

**Problem analysis — it walks by shuffling, and the clock never engaged.** Compare the per-term
breakdowns at 750k → 1.5M:

| term | 750k | 1.5M | reading |
|---|---|---|---|
| mean vx | 0.217 | **0.656** m/s | velocity tracking solved (now in the 0.5–1.0 command band) |
| `r_vel` | 0.660 | **1.336** | the optimizer's whole focus — 110 % of total reward |
| mean z | 0.250 | 0.269 m | **crouched** — well under the 0.33 target the whole time |
| `r_clearance` | 0.054 | **0.025** | feet lift *less* at 1.5M than at 750k |
| `r_airtime` | −0.013 | **−0.022** | **negative** — touchdowns happen before the 0.2 s air-time threshold; no flight phase |
| `r_phase` | 0.0031 | **0.0107** | 0.0107 / 0.15 ⇒ phase_reward 0.071 ⇒ `phase_match ≈ 0.54` — barely above the 0.5 standing baseline |

The diagnosis is unambiguous. As the velocity reward kicked in (0.66 → 1.34), the policy learned to
go faster **by shuffling lower, not by stepping** — `r_clearance` *dropped* and `r_airtime` stayed
negative. A fast, low, quick-tap shuffle maximizes `r_vel` while paying almost nothing for swing
arcs, so that's the attractor it found. And `phase_match ≈ 0.54` means the feet are essentially
uncorrelated with their phase clocks — **the entire Tier-2 mechanism is dormant**: the 4 frequency
actions and 8 phase observations are carrying almost no behaviour. The viewer confirms it: a stiff,
arcless strut. This is not a training failure (it's stable, high `expl_var`, full episodes) — it's a
**reward-shaping** failure: synced, arced stepping simply doesn't pay more than the cheap shuffle, so
the optimizer never leaves the shuffle basin.

**Proposed solution → Run 17:** raise the incentives that the shuffle is dodging, so real stepping
beats it. Three reward-weight changes (no action/obs change, so warm-start stays compatible):
`W_PHASE` **0.15 → 0.25** (make honoring the clock pay — wake the Tier-2 mechanism), `W_CLEARANCE`
**0.10 → 0.20**, and `FOOT_CLEAR_TARGET` **0.08 → 0.10** (force genuine swing arcs instead of skims).
If the strut persists, the next levers are: gate clearance to the swing half of each leg's phase
(`φ_i ≥ 0.5`) so arc + timing reinforce, raise `W_PHASE` further toward 0.4, or lift the crouch with
a stronger `r_height`. Trained **from scratch** so the new incentives shape the gait from the start
rather than inheriting the strut basin (warm-starting Run 16 + exploration boost is the cheaper
fallback if the from-scratch run is too slow).

**★ Note:** preserve Run 16's output (rename `checkpoints/` → e.g. `legacy_pmtg_strut/`) before
launching Run 17, which wipes `checkpoints/`. It's the "Tier-2 walks but struts" milestone and the
warm-start fallback source.

---

## Run 17 — Stronger stepping incentives (wake the phase clock, force swing arcs)

**Date:** 2026-06-14 · **Warm-start:** none (from scratch) · **Status:** PARTIAL — fixed what it
targeted, exposed the next problem · **Result:** the levers worked — the phase clock **engaged** and
the feet now **lift and arc** — but the gait **stumbles/straggles**: with nothing rewarding
inter-leg coordination, the four phases drifted off the trot relationship into an uncoordinated,
high-stepping pattern.

**Changed from previous (Run 16):** three reward-weight increases, all aimed at making synced, arced
stepping out-earn the shuffle Run 16 settled into — no action/observation change:
`W_PHASE 0.15 → 0.25` (per-leg phase consistency — give the policy a real reason to honor its clock),
`W_CLEARANCE 0.10 → 0.20` (swing-foot lift), `FOOT_CLEAR_TARGET 0.08 → 0.10 m` (the clearance reward
now saturates only at a 10 cm lift, so feet must actually arc, not skim).

**Key metrics (3M):** `ep_len_mean` **996**, `ep_rew_mean` **1950** (up from Run 16's 1400), `std`
0.494, `approx_kl` 0.0207, `clip_fraction` 0.279, `expl_var` 0.91.

**Problem analysis — the levers did exactly their job; the failure is on a new axis.** Run 16 (1.5M) →
Run 17 (3M):

| term | Run 16 | Run 17 | reading |
|---|---|---|---|
| `r_clearance` | 0.025 | **0.211** | 8× — feet lift and arc now (the strut is gone) |
| `r_phase` | 0.011 | **0.139** | 0.139 / 0.25 ⇒ `phase_match` 0.54 → **0.78** — the clock engaged |
| `r_airtime` | −0.022 | **−0.002** | ~0 — touchdowns now near the 0.2 s threshold (real flight phases) |
| mean z | 0.269 | **0.298** | body lifted toward the 0.33 target |
| `r_angvel` | −0.188 | −0.110 | less buck |
| mean vx | 0.656 | 0.711 | good forward speed |

Every term the Run 17 levers targeted improved — the strut diagnosis was right and the fix landed.
**But the gait looks worse**, and the breakdown says why: we reward *per-leg* behaviour and never
reward *inter-leg* coordination. `r_phase` scores each leg against **its own** clock (0.78 = each leg
honours its own phase), but nothing constrains the **relationship between the four phases**. With
fully independent per-leg frequencies, the clocks drifted off the diagonal-trot init into an
uncoordinated pattern, so the dog high-steps with each leg individually on-clock yet the four legs
out of trot with each other — feet up, body up, 0.71 m/s, and stumbling. The legacy fixed-diagonal
`r_gait` supplied exactly this coordination (Run 15's clean trot); pure per-leg self-consistency does
not. **This is the structural cost of full per-leg phase independence**, surfacing now that the feet
are actually lifting.

**Proposed solution → Run 18:** add an inter-leg **coordination prior** the per-leg design is
missing, gated by `(1−d)` so it shapes the calm gait but releases for retiming/recovery. Candidate
forms (decision pending): (a) a phase-space diagonal-coupling reward — pull the four phases toward
`FL≈RR`, `FR≈RL`, pairs offset 0.5 (pure Tier-2, keeps retiming); (b) reintroduce the proven
contact-based diagonal `r_gait` from Run 15, gated and *on top of* the policy-controlled cadence
(coordination from the prior, timing still from the policy); or (c) reduce frequency authority
(shrink `FREQ_RANGE`, raise `W_FREQ`) so the phases can't drift far from the trot init. Secondary:
trim `W_CLEARANCE` 0.20 → 0.15 if the high-stepping looks exaggerated.

**★ Note:** preserve Run 17's output (`checkpoints/` → e.g. `legacy_pmtg_run17/`) before Run 18 wipes
it — it's the "clock + arcs working, coordination missing" milestone and a warm-start source.

---

## Run 18 — Tier-3 PMTG: Foot Trajectory Generator + residuals (from scratch)

**Date:** 2026-06-15 · **Warm-start:** none (new FTG+residual policy — from scratch) ·
**Status:** PARTIAL — strut and stumble both fixed; coordination still not diagonal · **Result:** the
FTG works — feet lift and arc, the gait is stable and quick — but the footfall is **not a trot**:
FL+RL land together, then RR a beat later, then FR (an ipsilateral/lateral-ish pattern). The trot
*init* didn't survive; the legs drifted off it via the frequency channel. Preserved in
`checkpoints_weird_gait/`.

**Changed from previous (Run 17):** stopped trying to make coordination and the swing arc *emerge*
from rewards and made both **structural**, following Lee 2020 / Iscen 2018 (see §1.7). Three changes,
no action/obs shape change (still 16-D / 60-D): (1) **shared base clock + fixed trot offsets** — the
four phases ride one `BASE_FREQ = 1.5 Hz` clock initialized to `[0,0.5,0.5,0]`, policy adds only a
bounded per-leg frequency offset, so the legs can't drift apart (fixes Run 17's stumble); (2) a
**joint-space Foot Trajectory Generator** adds a nominal thigh-lift + knee-flex swing arc
(`q_target = default + FTG(φ) + 0.5·residual`), so the foot steps by construction (fixes Run 16's
strut); (3) the 12 joint dims are now **residuals on top of the FTG**, and `r_phase`/`r_clearance`
drop to auxiliary weights (`W_PHASE 0.25→0.15`, `W_CLEARANCE 0.20→0.10`) since the FTG supplies the
arc + coordination. Bundled migration run, from scratch.

**Pre-flight verification (no training yet):** by forward kinematics the FTG produces a nominal peak
foot lift of **0.099 m** (target ~0.10) on the correct (upward) arc, zero during stance; over 80
control steps with zero policy output the four phases stay locked at the trot relationship
`[0, 0.5, 0.5, 0]`; a `−1` frequency action stalls a leg's clock (f → 0); action 16-D / obs 60-D and
the reward terms sum correctly. So the structure behaves before a single PPO step.

**Key metrics (3M):** `ep_len_mean` **1000**, `ep_rew_mean` **1830**, `std` 0.488, `approx_kl` 0.022,
`clip_fraction` 0.264, `expl_var` **0.993** (settled, confident). Breakdown @3M: `r_vel` 1.437
(81 %, mean vx **0.707**), `r_height` 0.345 (mean z **0.284**), `r_clearance` 0.104, `r_phase`
**0.112**, `r_freq` **−0.036**, `r_jointvel` −0.139, `r_angvel` −0.084.

**Problem analysis — the FTG fixed both prior failures, but coordination still drifted.** The strut
(Run 16) and the uncoordinated *non-stepping* (Run 17) are gone: `r_clearance` is healthy (0.104, the
FTG lifts the feet), `r_phase` is **0.112 / 0.15 ⇒ phase_match ≈ 0.87** (the feet faithfully follow
their own clocks), height recovered toward target, and episodes run the full 1000 steps. **But the
gait is not diagonal** — FL+RL together, then RR, then FR. Why: `r_freq` −0.036 ⇒ mean `Σ a_freq² ≈
1.8`, i.e. the policy is actively running the per-leg frequency channel, and because Run 18 *integrated*
that frequency (`phase_i += (BASE_FREQ + FREQ_RANGE·a_i)·dt`) the relative phases drift without bound;
over a 20 s episode even a tiny persistent `a_i` walks the legs off the trot offset into a different
phase-lock. The feet then dutifully follow the *drifted* phases (match 0.87), so the gait is internally
consistent but laterally coordinated, not diagonal. **The trot init is a starting point, not a
constraint** — exactly the Tier-2 drift, just slower because the shared clock + FTG damp it. The
`W_FREQ`/init structure was not enough to *hold* the diagonal.

**Proposed solution → Run 19:** keep the per-leg frequency mechanism (it's the ANYmal point — needed
for terrain retiming / leg stall) but make coordination a **hard bound**: accumulate the frequency
action into a per-leg phase offset `delta_i` that is **clamped to ±`DELTA_MAX` (0.15 cycle)** off the
trot slot and **bled back to 0 when calm** (`RECENTER`, `d`-gated). The clamp makes diagonal drift
impossible; the recenter returns flat walking to a clean trot; the frequency channel (and thus
retiming/stall up to the bound) stays alive and opens up under disturbance / on terrain. No
action/obs shape change. (Reward-term trims — e.g. dropping the now-redundant `r_airtime` and the
FTG-fighting `r_pose` — are on the table but deferred to Ved's pick; not bundled into Run 19.)

**★ Note:** Run 18 preserved at `checkpoints_weird_gait/` — the "FTG works, coordination drifts"
milestone.

---

## Run 19 — Clamped integrated per-leg phase offset (pin the trot, keep retiming)

**Date:** 2026-06-15 · **Warm-start:** none (from scratch) · **Status:** PARTIAL — coordination
bounded, but stance + timing still unsatisfying · **Result:** the clamp stopped the lateral drift (FL
and RL are forced ~0.5 apart now, so it no longer collapses to the ipsilateral pattern), but the gait
is still **crouched** (~0.28 m) and the **trot timing is skewed** — the policy saturates the offset
clamp, so it's a constrained near-trot, not a clean one.

**Changed from previous (Run 18):** the per-leg phase is no longer a free-integrating frequency.
A shared global clock advances at `BASE_FREQ`; each leg's phase is `(global + TROT_OFFSET_i +
delta_i) % 1`, where the frequency action *accumulates* `delta_i` (`delta_i += FREQ_RANGE·a_i·dt`,
so `a_i` still sets each leg's phase rate — the ANYmal mechanism, can stall/advance a leg) but
`delta_i` is **hard-clamped to ±`DELTA_MAX = 0.15` cycle** and, when calm, **recentered toward 0**
(`delta_i *= 1 − RECENTER·(1−d)`, `RECENTER = 0.02`). So the diagonal can't drift away (hard bound),
flat walking settles back to a clean trot (recenter), and per-leg retiming stays fully available and
opens up under disturbance / on terrain (`d`-gate releases the recenter). `W_FREQ` demoted to a small
anti-churn penalty (the clamp/recenter do the coordination work). No action/obs shape change.

**Pre-flight note:** the FTG + dims + reward-sum were runtime-verified on the Run 18 base; the clamp/
recenter changes were verified by code inspection (the workspace sandbox's view of `train.py` was
stuck on a stale cache this round, so the numeric `|delta| ≤ 0.15` check is deferred).

**Key metrics (3M):** `ep_len_mean` **1000**, `ep_rew_mean` **1810**, `std` 0.457, `approx_kl` 0.019,
`clip_fraction` 0.244, `expl_var` 0.993. Breakdown @3M — essentially identical to Run 18: `r_vel`
1.431 (mean vx **0.701**), `r_height` 0.326 (mean z **0.281**), `r_clearance` 0.108, `r_phase`
**0.117** (match ≈0.89), `r_freq` **−0.035**, `r_jointvel` −0.134, `r_angvel` −0.079.

**Problem analysis — clamp held coordination, but the gait is crouched and the policy fights the
trot.** The clamp did its job: the lateral FL+RL drift is gone (the deltas can't carry a leg more than
0.15 off its trot slot). But two complaints remain (Ved, from the viewer): it sits too low, and the
trot timing is still off. The breakdown explains both. (1) **Crouch:** mean z 0.281 vs the 0.33 target
— and the robot's *default* stance height is only 0.27 m (home keyframe), so 0.33 requires actively
extending the legs out of default. Three things prevent that: `RESIDUAL_SCALE = 0.5` (limited
authority to push the stance up), `r_pose` (penalizes deviation from the 0.27 default — actively pulls
it back down *and* fights the FTG swing), and a weak height reward (weight 0.5, ±0.05 tolerance, so
5 cm low costs ~0.2). (2) **Timing:** `r_freq` −0.035 ⇒ rms `a_freq ≈ 0.66`, i.e. the policy is
driving the offset action hard and the deltas sit pinned at the ±0.15 clamp edge — it actively *wants*
to leave the trot. Likely root cause: at the crouched low COM a true 2-beat diagonal trot is unstable,
so the policy skews toward a statically-stabler walk-like timing and the clamp only half-holds it.
i.e. **the crouch is probably *causing* the bad timing** — fix the height and the trot should settle.

**Proposed solution → Run 20:** raise the stance and let the trot settle. `RESIDUAL_SCALE 0.5 → 0.7`
(authority to extend the stance), height reward weight `0.5 → 0.8` (motivate standing tall), **drop
`r_pose`** (it pulls toward the 0.27 default and fights the FTG) and **drop `r_airtime`** (redundant —
the FTG already gives ~0.33 s of swing, so it sits at 0), and tighten `DELTA_MAX 0.15 → 0.10` (less
room to skew the timing while the height fix does the real work). Hypothesis: height is the root cause
and the trot + timing clean up as a consequence; if `r_freq` stays saturated after, the 1.5 Hz cadence
itself is wrong for the speed and we tackle that next.

**★ Note:** Run 19 preserved at `checkpoints_weird_gait_run19/` (rename `checkpoints/` before Run 20
wipes it).

---

## Run 20 — Stand tall, let the trot settle (raise stance + trim fighting rewards)

**Date:** 2026-06-15 · **Warm-start:** none (from scratch) · **Status:** FAIL — regressed · **Result:**
**worse than Run 19**: the legs *stutter* and stop tracking the FTG, and the body is still crouched
(actually lower). Two of the five changes backfired.

**Changed from previous (Run 19):** five coupled edits aimed at standing tall: `RESIDUAL_SCALE 0.5 →
0.7`, height-reward weight `0.5 → 0.8` (`W_HEIGHT`), **drop `r_pose`**, **drop `r_airtime`**,
`DELTA_MAX 0.15 → 0.10`.

**Problem analysis — `r_pose`-drop + residual-raise unleashed the policy; the height lever was wrong.**
Run 19 (3M) → Run 20 (1.5M, first two checkpoints):

| term | Run 19 | Run 20 @1.5M | reading |
|---|---|---|---|
| `r_jointvel` | −0.134 | **−0.251** | joint thrash ~doubled |
| `r_angvel` | −0.079 | **−0.231** | torso buck ~tripled |
| `r_phase` → match | 0.117 (0.89) | **0.057 (0.69)** | feet stopped tracking the FTG |
| mean z | 0.281 | **0.263** | crouch unchanged / slightly worse |

Dropping `r_pose` (the residual regularizer) **and** simultaneously giving the residuals +40 % authority
(`RESIDUAL_SCALE 0.7`) let the policy override the FTG with big jittery corrections instead of riding
it — hence the stutter (jointvel/angvel up) and the collapse in phase tracking (0.89 → 0.69). And the
height lever simply didn't work: `W_HEIGHT` up + more authority did **not** raise the body (0.263). So
the crouch is **not** an authority/pose problem — Runs 18–19 had `r_pose` on and low residual and still
sat at ~0.28. The robot just prefers ~0.28 in this FTG-trot. Forward-kinematics check: the *default
stance* itself only stands at **0.288 m** (and a true trot has just 2 legs supporting), so 0.33 was
never reachable by reward pressure — wrong lever entirely.

**Proposed solution → Run 21:** (1) **revert the backfire** — `RESIDUAL_SCALE 0.7 → 0.5` and **restore
`r_pose`** (it tidied the gait and was not the crouch cause); keep `DELTA_MAX 0.10`, `r_airtime`
dropped. (2) **Raise height STRUCTURALLY, not via reward** — add a constant *stance-lift* to the
nominal pose so the legs stand straighter by default; FK-tuned to thigh `−0.10` / calf `+0.20` rad,
which lifts the nominal standing height **0.288 → 0.320 m**. This directly addresses Ved's theory: while
tracking the fixed FTG the policy never got to vary the motors enough to lift itself, so we bake the
lift into the nominal stance and let the FTG swing from a taller base (and `r_pose` now references the
taller pose, so it no longer fights height).

**★ Note:** preserve Run 20 if wanted (rename `checkpoints/`); it's the "don't drop the regularizer
while raising authority" cautionary milestone.

---

## Run 21 — Revert backfire + structural stance-lift (stand tall the right way)

**Date:** 2026-06-15 · **Warm-start:** none (from scratch) · **Status:** PARTIAL — height + stutter
fixed; new issue: rear legs barely lift · **Result:** the revert + structural stance-lift **worked** —
the body stands taller (~0.29–0.30, up from ~0.26) and the Run-20 stutter is gone — but the **rear two
legs barely lift** (front legs follow the FTG, rears stay low). Preserved in `frontnback/`.

**Changed from previous (Run 20):** (1) **revert** `RESIDUAL_SCALE 0.7 → 0.5` and **restore `r_pose`**
(W_POSE 0.02) to kill Run 20's stutter / FTG-override; (2) **structural stance-lift** — the nominal
standing pose is raised by adding `STANCE_LIFT` (thigh `−0.10`, calf `+0.20` rad per leg) to the
default joints, so the robot stands at ~0.32 m *by construction* rather than fighting to lift. The
FTG swing and `r_pose` now both reference this taller nominal. Kept from Run 20: `DELTA_MAX 0.10`,
`r_airtime` dropped, `W_HEIGHT 0.8`. No action/obs shape change.

**Pre-flight (FK):** default nominal stands at 0.288 m; with the stance-lift the nominal stands at
**0.320 m**.

**Key metrics (1.5M):** mean z **0.292** (was ~0.26 in Runs 19–20 — structural lift confirmed),
`r_jointvel` −0.134 / `r_angvel` −0.113 (back to Run-19 levels — stutter gone), `r_phase` 0.093
(match ≈0.81). The two intended fixes landed.

**Problem analysis — height/stutter solved; rear legs don't lift (new).** Viewer: front legs follow
the FTG arc, **rear two legs barely leave the ground**. FK check: the FTG lifts all four legs
**identically (0.094 m each)** — so it is NOT geometry. The rear-low is **learned**: the policy cancels
the rear FTG with an opposing residual because keeping the (propulsive) rear legs low and long in
stance maximizes `r_vel`, and it pays no price for it — `r_phase` only needs the foot to break contact
(a sliver of air counts) and `r_clearance` is an optional bonus it can decline.

**Proposed solution → Run 22/23:** make following the FTG a *requirement*, not a bonus. (Run 22 first
tried the cheap thing — raise `W_CLEARANCE` — which failed; Run 23 is the real fix, an FTG-tracking
shortfall penalty. See below.)

---

## Run 22 — W_CLEARANCE 0.1 → 0.2 (cheap attempt, FAILED)

**Date:** 2026-06-15 · **Warm-start:** none · **Status:** FAIL — no improvement · **Result:** rears
still barely lift, and now a visible **asymmetry between the two front legs** too.

**Changed from previous (Run 21):** raised `W_CLEARANCE` 0.1 → 0.2 (Ved's edit), hoping a bigger lift
bonus would pull the rear legs up.

**Problem analysis — you can't make an optional bonus mandatory by enlarging it.** At 750k `r_clearance`
did rise (0.095 vs Run 21's 0.039 — the bonus is bigger) but the behaviour didn't change: the policy
still declines it on the rears because the propulsion gain from a low rear stance outweighs the larger
forgone bonus. And since nothing *requires* any specific per-leg clearance, the fronts drifted to
**unequal** lift too. Confirms the Run 21 diagnosis: `r_clearance` is a reward that can be declined;
the FTG is additively cancellable. Neither *forces* a leg to follow the swing.

**Proposed solution → Run 23:** an FTG-tracking **shortfall penalty** (below).

---

## Run 23 — FTG-tracking shortfall penalty (make every leg follow the swing)

**Date:** 2026-06-15 · **Warm-start:** none (from scratch) · **Status:** PARTIAL — helped, not solved ·
**Result:** the tracking penalty pulled the rears up from ~0 to **~0.035 m**, but they're still about
**half** the fronts (~0.072 m). The policy *pays* `r_track` (−0.088) rather than fully lift — propulsion
still wins. (Per-leg logging now lets us measure this; next we check duty factor to see if the rear-low
is legitimate propulsion.)

**Changed from previous (Run 22):** (1) **new `r_track`** — a per-leg **shortfall** penalty: each step,
compute the FTG's nominal foot clearance for each leg (`FTG_FOOT_LIFT·lift(phase_i)`, 0 in stance,
peak ~0.09 mid-swing) and penalize only the amount the actual foot falls **below** it:
`−relax·W_TRACK·Σ_i max(0, nominal_i − actual_i)`, `W_TRACK = 3.0`. This makes following the FTG swing a
*requirement* (a cost for not lifting), not a decline-able bonus, and it's symmetric across legs (fixes
both the rear-low and the front asymmetry). (2) **reset `W_CLEARANCE` 0.2 → 0.1** (the tracking penalty
now does the real work). (3) **per-leg swing-clearance logging** added to the breakdown (`swc_*`/`sw_*`
→ mean swing clearance per leg) so we can *measure* rear vs front lift instead of eyeballing it.
Everything else from Run 21 kept.

**Why this still allows terrain deviation (the key design point).** `r_track` is **shortfall-only** and
**`d`-gated**. Shortfall-only = lifting *higher* than the FTG nominal costs nothing, so on terrain the
policy is free to raise a foot to clear a bump for free — only *flattening* a swing is penalized.
`d`-gated = under disturbance/instability the penalty releases, so recovery footwork and a foot-trapping
retract (foot held below the arc after snagging) aren't fought. And because it's a *soft* penalty, not
a hard constraint, the large survival/progress rewards on terrain can still outweigh it when a genuine
deviation is needed — the FTG stays a *breakable prior*. On flat ground there's no countervailing
benefit to flattening, so the penalty wins and every leg follows the swing.

**Pre-flight:** tracking math verified standalone — a flat rear swing yields shortfall 0.18 → `r_track`
−0.54; lifting to nominal → 0; lifting *above* nominal → 0 (deviation free). Per-leg log formatter
verified. (Full env run deferred while the sandbox mount cache for `train.py` is stale.)

**Key metrics (1.5M):** `r_vel` 1.325 (**74.7 %** — still dominates), vx 0.650; `r_height` 0.644, z
**0.294** (height holding); `r_track` **−0.088** (policy paying the penalty); `r_phase` 0.108 (match
≈0.86). **Per-leg swing clearance: FL 0.071, FR 0.074, RL 0.035, RR 0.037** — rears ~half the fronts.

**Problem analysis — the tracking penalty works but is outgunned by velocity.** `r_track` did its job
directionally (rears 0→0.035) but the policy chose to eat the −0.088 cost rather than lift fully,
because the propulsion gain a low rear buys in `r_vel` (75 % of the reward) exceeds it. So the rear-low
is an *equilibrium* between the propulsion prize and the tracking cost. Mechanistically (see the
discussion logged with this run): forward velocity = pushing backward on the ground; only stance legs
push; the rear legs are the push-off legs, so keeping them low/in-contact longer maximizes propulsive
contact time, while the fronts lift freely because they must swing forward to reach the next foothold.
The front-lifts/rears-don't asymmetry is the fingerprint of this fore-aft division of labour.

**Proposed solution → Run 24:** two moves. (1) **Rebalance the saturated primary rewards** — `r_vel`
1.5 → 1.0 and `W_HEIGHT` 0.8 → 0.5 (both down together to keep ~2:1 vel:height so it still walks and
doesn't "stand tall, won't walk"). Lowering the velocity prize directly shrinks the propulsion
incentive that holds the rears down. (2) **Add per-leg DUTY-FACTOR logging** (stance fraction) to *test
the propulsion hypothesis*: if the rears show a markedly higher duty factor than the fronts, the
rear-low gait is **legitimate propulsion**, not laziness — in which case we may accept front-high /
rear-low rather than force symmetry (caveat: low rears will clip on terrain, so we'll likely still want
some rear clearance for Phase-4 terrain). `W_TRACK` left at 3.0 for now — decide whether to push it
after the duty data is in.

**★ Preserved:** Run 23 in `checkpoints/`-of-record (rename before Run 24 wipes it).

---

## Run 24 — Rebalance velocity/height + duty-factor logging (test the propulsion hypothesis)

**Date:** 2026-06-15 · **Warm-start:** none (from scratch) · **Status:** COMPLETE — rebalance neutral,
**propulsion hypothesis REFUTED** · **Result:** the velocity/height rebalance behaved as predicted but
did **not** fix the rear-low, and the duty-factor data shows the rears aren't propelling — they're just
**under-lifting**. So we do *not* accept the asymmetry; the amplitude fix (Run 25) is the right path.

**Changed from previous (Run 23):** (1) **`r_vel` 1.5 → 1.0** (`W_VEL`) and **`W_HEIGHT` 0.8 → 0.5** —
the two primary terms were saturating the gradient; cut together to keep the vel:height ratio (~2:1).
(2) **Per-leg duty-factor + swing-clearance logging** added. `W_TRACK` unchanged (3.0).

**Key metrics:** mean vx **0.556** (down from Run 23's 0.65 — lighter velocity weight, still walking),
mean z **0.286** (held despite the lighter height reward — structural stance-lift doing the work).
Per-leg readout:

| leg | swing clearance (m) | duty factor (stance frac) |
|---|---|---|
| FL | 0.0615 | 0.643 |
| FR | 0.0596 | 0.633 |
| RL | **0.0390** | 0.620 |
| RR | **0.0377** | 0.637 |

**Problem analysis — the duty factor refutes "propulsion", so the rear-low is plain under-lifting.**
Two clean results. (1) The rebalance was *neutral* on the gait: cutting `r_vel` slowed it (0.65→0.556,
still in band) and cutting `r_height` did **not** drop the body (0.286, height is structural now — good
confirmation), but neither moved the rear clearance (RL/RR still ~0.038, same as Run 23). So reducing
the velocity prize did **not** fix the rears — evidence against "they stay low for velocity/propulsion."
(2) The decisive one: **all four duty factors are ~equal (0.62–0.64)** — RL is even the *lowest*. If the
rears were propelling, they'd spend *more* time in stance (higher duty); they don't. So the rears are
**not** push-dominant support legs — they spend the same fraction of the cycle on the ground as the
fronts but simply lift ~40 % less when they swing. **Conclusion: the rear-low is unconstrained
under-lifting, not a legitimate asymmetric gait.** My earlier propulsion story was wrong; the data
killed it. (Front-front is nearly symmetric now: 0.0615 vs 0.0596.)

**Proposed solution → Run 25:** since it's under-lifting (not propulsion), the fix is direct amplitude
pressure with a *shared per-leg target* — **restore `r_airtime` as a peaked target** (Gaussian around a
common flight time, pulls every leg to the same swing duration) **+ keep `r_track`** (shared FTG arc
height). This is what Run 8 had (airtime) + what ANYmal does (per-leg clearance). If the rears still lag
after, raise `W_AIRTIME`/`W_TRACK`; if a visible asymmetry survives reward shaping, mirror augmentation
(Yu 2018) is the guaranteed fallback.

---

## Run 25 — Restore r_airtime as a peaked target (amplitude + symmetry)

**Date:** 2026-06-15 · **Warm-start:** none (from scratch) · **Status:** SUCCESS — flat Tier-3 trot
essentially solved · **Result:** the rears came **up to the FTG arc** and the gait is much more
symmetric — `r_track` collapsed −0.094 → **−0.007** (legs now follow the arc), `r_phase` match ~**0.92**,
buck way down (`r_angvel` −0.116 → **−0.033**), vx **0.724**. Residual: a small front/rear clearance gap
(fronts lift slightly above the 0.09 m arc, rears sit at it) — cosmetic, not a failure. **Preserved as
`checkpoints_ftg/` and set as the warm-start source for `--push`/`--terrain`.**

> **★ Folder renamed (2026-06-15):** this Run-25 policy was the `checkpoints_ftg/` folder. During the
> repo reorg it was **renamed `checkpoints_ftg/` → `run25_ftg/`** and moved under
> `legacy/legacy_16D_output/run25_ftg/` (alongside `run26_push_ftg/`). It is the clean flat Tier-3 trot
> and the warm-start source for the push line (Run 26). Older references to `checkpoints_ftg/` in the
> logs/handoff all mean this folder.

**Changed from previous (Run 24):** **restore `r_airtime`**, but as a **peaked target** instead of the
legacy unbounded linear: at each touchdown a leg scores `exp(−(flight_time − AIR_TIME_TARGET)² /
2·AIR_TOL²)` (target 0.25 s, tol 0.08, weight `W_AIRTIME = 0.5`). Because the target is the **same for
all four legs**, it pulls every leg to the *same* swing duration → directly fights both the rear-low
(a dragging rear has too-short flight → low reward) and the front-front mismatch (both fronts pulled to
the same flight time). Peaked (not linear) so it can't be gamed by ever-longer hops. `r_track` (0.09 m
FTG-clearance shortfall) kept — airtime governs swing *duration*, track governs swing *height*; both
shared across legs. `r_clearance` left at 0.10 (now partly redundant). Per-leg duty + clearance logging
now active. Rationale mirrors Run 8 (airtime gave its symmetric amplitude) and Lee 2020 (per-leg
clearance reward + target-smoothness).

**Key metrics (3M):** `r_vel` 0.959 (62 % — no longer saturating, was 75 %), vx **0.724**; `r_height`
0.461; `r_airtime` **0.052** (new term active); `r_clearance` 0.091; `r_phase` **0.126** (match ~0.92);
`r_track` **−0.007** (was −0.094 — the rears are now lifting to the arc); `r_angvel` **−0.033** (was
−0.116 — much smoother); `r_jointvel` −0.122. TOTAL 1.54.

**Problem analysis — the amplitude fix worked; flat is essentially done.** The peaked `r_airtime` +
`r_track` (both shared per-leg targets) pulled the rears up to the FTG arc: `r_track` near zero means
no leg is materially below the 0.09 m nominal, and `r_phase` 0.92 + `r_angvel` collapsing show a clean,
coordinated, low-buck trot. The only residual is a small front/rear clearance gap, which is `r_track`
being **shortfall-only** by design (fronts may exceed the arc for free; rears sit at it) — wanted for
terrain, cosmetic on flat. Net: stance height ✓, coordination ✓, amplitude/symmetry ✓ (near-uniform).
The clearance *reward* (`r_clearance`) stayed negligible/auxiliary throughout — `r_track` (the FTG-arc
floor) is what set amplitude, exactly as intended.

**Proposed solution → next:** **flat Tier-3 is done** — preserved as `checkpoints_ftg/`. Two directions:
(a) if the residual front/rear gap matters, raise the **FTG arc** (`FTG_THIGH_AMP`/`FTG_CALF_AMP` +
`FTG_FOOT_LIFT`, FK-tuned to ~0.12 m) so the enforced floor rises for all legs; (b) otherwise proceed —
**warm-start `--push`** from `checkpoints_ftg/` (re-acquire push recovery on the Tier-3 architecture),
then **`--terrain`** (widen `DELTA_MAX`/FTG arc). `WARM_START` now points at `checkpoints_ftg/`.

---

<!-- TEMPLATE — copy below and fill in for the next run.

## Run N — <short name>

**Date:** · **Warm-start:** · **Status:** · **Result:**

**Changed from previous:** (one conceptual change)

**Problem analysis:** (key metrics: ep_rew_mean, ep_len_mean, std, approx_kl, clip_fraction,
expl_var; the per-term reward breakdown numbers + arithmetic; viewer behaviour; eval numbers if run)

**Proposed solution → Run N+1:**

-->
