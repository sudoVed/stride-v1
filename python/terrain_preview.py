"""Preview the terrain WITHOUT training. Edit TERRAIN_OCTAVES / TERRAIN_* / foot radius, then:

    python terrain_preview.py [terrain_height]      # default height 0.10

Reports how much the terrain varies (elevation), how FAST it changes (mean slope = the derivative you
asked about), and the worst adjacent-tile step vs the foot's bridging size -- so you know before
training whether feet will catch. It calls the REAL _terrain_heights() from train.py, so it always
matches what training will use.
"""
import os
import sys
import importlib.util
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("go2_train", os.path.join(_here, "train.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

TH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10     # terrain height to preview
FOOT_R = 0.035                                             # foot radius in go2.xml (approx step it bridges)
P = T.TERRAIN_PITCH

xs = np.arange(T.TERRAIN_XB, T.TERRAIN_XF + 1e-6, P)
ys = np.arange(-T.TERRAIN_YH, T.TERRAIN_YH + 1e-6, P)
X, Y = np.meshgrid(xs, ys)
xy = np.stack([X.ravel(), Y.ravel()], axis=1)

rng_rng, rng_std, rng_slope, rng_maxstep = [], [], [], []
for k in range(100):
    H = T._terrain_heights(xy, np.random.default_rng(k), TH).reshape(len(ys), len(xs))
    dx = np.abs(np.diff(H, axis=1))
    dy = np.abs(np.diff(H, axis=0))
    rng_rng.append(H.max() - H.min())
    rng_std.append(H.std())
    rng_slope.append(np.degrees(np.arctan((dx.mean() + dy.mean()) / 2.0 / P)))   # mean slope = derivative
    rng_maxstep.append(max(dx.max(), dy.max()))

p99 = np.percentile(rng_maxstep, 99)
print("octaves (period_m, amp) : {}".format(T.TERRAIN_OCTAVES))
print("height {:.0f} cm | pitch {:.0f} cm | foot bridge ~{:.0f} mm | strip {:.0f}x{:.0f} m".format(
    TH * 100, P * 100, FOOT_R * 1000, T.TERRAIN_XF - T.TERRAIN_XB, 2 * T.TERRAIN_YH))
print("-" * 60)
print("elevation range : {:.1f} cm   (how much it varies)".format(np.mean(rng_rng) * 100))
print("elevation std   : {:.1f} cm   (spread of the distribution)".format(np.mean(rng_std) * 100))
print("mean slope      : {:.1f} deg  (how FAST it changes = the derivative)".format(np.mean(rng_slope)))
print("max adj. step   : mean {:.0f} mm, p99 {:.0f} mm".format(np.mean(rng_maxstep) * 1000, p99 * 1000))
print("-" * 60)
# With mocap tiles contact is correct, so steps no longer CLIP -- they're just difficulty. Two thresholds:
# under the foot radius = feet roll over effortlessly; under ~leg lift = the policy must step over it.
EASY = FOOT_R          # ~35 mm: effortless roll-over
HARD = 0.065           # ~leg lift: above this the robot likely can't clear it -> face-plant
if p99 < EASY:
    print("DIFFICULTY: easy   -- feet roll over every step.")
elif p99 < HARD:
    print("DIFFICULTY: medium -- feet clear most; policy must actively step over the steeper bits (good).")
else:
    print("DIFFICULTY: hard   -- steps up to {:.0f} mm exceed comfy leg lift (~{:.0f} mm); may face-plant.".format(
        p99 * 1000, HARD * 1000))
    print("            lower terrain-height, drop TERRAIN_PITCH (re-run make_box_terrain.py), or grow the foot.")
