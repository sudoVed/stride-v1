"""Generate scene_terrain.xml as a FORWARD STRIP of overlapping BOX tiles (mocap bodies).

Two reasons for the design:
 - MOCAP bodies (not static geoms): MuJoCo bakes the static-geom collision BVH at compile time, so
   moving static tiles leaves their collision behind -> feet sink into raised tiles. Mocap bodies get
   live collision; train.py sets each tile height per episode via data.mocap_pos.
 - FORWARD STRIP (long in +x, narrow in y): the robot only walks +x, so a symmetric square wastes half
   its tiles behind it AND isn't long enough forward (robot walks off the edge). A strip puts the tiles
   where they're used: x in [XB, XF], y in [-YH, YH]. Must match TERRAIN_* in train.py.

Tiles overlap (no gaps, #2307) and use contype=1 conaffinity=2 (collide the robot, not each other).
Run once:  python make_box_terrain.py
"""
import os

XB, XF, YH, PITCH = -1.5, 18.0, 4.0, 0.5     # forward strip extent (match train.py TERRAIN_*)
HALF_XY = 0.30                                # tile half-width: 0.60 wide on 0.50 pitch -> 0.10 overlap
HALF_Z = 0.50                                 # tile half-height (1 m pillars, mostly buried); top = z + HALF_Z


def frange(lo, hi, step):
    n = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, 3) for i in range(n)]


xs, ys = frange(XB, XF, PITCH), frange(-YH, YH, PITCH)
tiles, k = [], 0
for y in ys:                                  # row -> y (outer, row-major)
    for x in xs:                              # col -> x
        tiles.append(
            '    <body name="tile_{k}" mocap="true" pos="{x:.3f} {y:.3f} {pz:.3f}">\n'
            '      <geom type="box" size="{hx} {hx} {hz}" group="0" material="groundplane" '
            'friction="0.8 0.02 0.01" contype="1" conaffinity="2"/>\n'
            '    </body>'.format(k=k, x=x, y=y, pz=-HALF_Z, hx=HALF_XY, hz=HALF_Z))
        k += 1

xml = '''<mujoco model="go2 scene box-terrain">
  <include file="go2.xml"/>

  <statistic center="0 0 0.1" extent="0.8"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-130" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <!-- Forward strip: {nx}x{ny} = {nn} overlapping box tiles (mocap). x in [{xb},{xf}] m, y +/-{yh} m.
         train.py sets each tile height per episode via data.mocap_pos[mocapid, 2]. -->
{tiles}
  </worldbody>
</mujoco>
'''.format(nx=len(xs), ny=len(ys), nn=len(tiles), xb=XB, xf=XF, yh=YH, tiles="\n".join(tiles))

out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "mujoco_menagerie", "unitree_go2", "scene_terrain.xml"))
with open(out, "w") as f:
    f.write(xml)
print("wrote {}  ({} tiles = {}x{}, forward {}..{} m, lateral +/-{} m)".format(
    out, len(tiles), len(xs), len(ys), XB, XF, YH))
