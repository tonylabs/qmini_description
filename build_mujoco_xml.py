#!/usr/bin/env python3
"""Convert Qmini URDF -> a MuJoCo body-tree include (qmini_description.xml).

Emits a ``<mujocoinclude>`` holding ONLY the robot body tree (base_link free body +
the two leg chains), meant to be pulled into a world with
``<include file="qmini_description.xml"/>`` inside <worldbody>. The rest of the model
— compiler + mesh assets (dependencies.xml), scene/floor/lights (scene.xml),
actuators + sensors (world.xml) — lives in the sibling include files.

What this adds beyond a raw MuJoCo URDF compile:

  1. a real ``base_link`` BODY with a <freejoint> (the raw import welds the root link
     into worldbody, so there's no floating base / no body named base_link),
  2. base_link inertial recovered from the URDF (the weld drops it),
  3. an "imu" site at the pose of the URDF "imu" fixed joint (imu_link mount) so the
     gyro/accelerometer frames (declared in world.xml) match training,
  4. per-joint <joint armature=...> (reflected rotor inertia; policy is sensitive to
     it on the ankle),
  5. collision geoms set contype=1 conaffinity=0 -> robot geoms collide with the floor
     (conaffinity=1 in scene.xml) but never each other (convex-hull STLs interpenetrate
     at the bent-knee pose); visual geoms (group 1) stay non-colliding.

Joint names / axes / limits / link poses all come from urdf/qmini_description.urdf.
Re-run after any URDF change.
"""

import math
import os
import re
import xml.etree.ElementTree as ET
import mujoco

# run from this script's directory (= repo root) so the paths below stay
# relative & portable
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
URDF = "urdf/qmini_description.urdf"
OUT = "mjcf/qmini_description.xml"

# default home height of the free base (init_state.pos[2]). Qmini's straight-leg
# foot bottom sits 0.4363 m below base_link, so 0.44 puts the feet just at the
# z=0 floor used by scene.xml / world_terrain.xml.
SPAWN_Z = 0.44

# joint-name prefix -> armature (reflected rotor inertia). Qmini joints are named
# "<part>_<l|r>" (e.g. hip_yaw_l) and the chain ends at ankle_pitch (no ankle_roll).
# hip_yaw/roll/pitch/knee share 0.02; ankle 0.0042 (values carried over from
# dreambo_asymmetry).
ARMATURE = {
    "hip_yaw":     0.02,
    "hip_roll":    0.02,
    "hip_pitch":   0.02,
    "knee_pitch":  0.02,
    "ankle_pitch": 0.0042,
}


def armature_for(joint_name):
    return ARMATURE[joint_name.rsplit("_", 1)[0]]  # strip the trailing _l/_r


def rpy_to_quat(roll, pitch, yaw):
    """URDF extrinsic-XYZ rpy -> MuJoCo (w, x, y, z) quaternion string."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return f"{w:g} {x:g} {y:g} {z:g}"


# --- 1. raw URDF -> MJCF via MuJoCo's compiler (gives bodies/joints/inertias/geoms) ---
with open(URDF) as f:
    urdf = f.read().replace("package://qmini_description/", "")
# inject a <mujoco><compiler meshdir=...> right after the opening <robot ...> tag.
# absolutize for this transient compile (MuJoCo resolves a relative meshdir vs the
# URDF file dir, not cwd). The emitted include carries no compiler/assets — the mesh
# names it references are defined in dependencies.xml.
mj_block = (
    f'\n  <mujoco><compiler meshdir="{ROOT}" '
    'balanceinertia="true" discardvisual="false"/></mujoco>'
)
urdf = re.sub(r"(<robot\b[^>]*>)", r"\1" + mj_block, urdf, count=1)
tmp = "urdf/_tmp.urdf"
with open(tmp, "w") as f:
    f.write(urdf)
try:
    m = mujoco.MjModel.from_xml_path(tmp)
    raw = "urdf/_raw.mjcf"
    mujoco.mj_saveLastXML(raw, m)
finally:
    os.remove(tmp)

tree = ET.parse(raw)
root = tree.getroot()
os.remove(raw)

# --- 2. restructure worldbody: weld-base -> base_link body + freejoint ---
# The raw import welds base_link (and its fixed-joint child imu_link) into worldbody,
# so its geoms sit as direct children of <worldbody> and the leg chains are separate
# <body> subtrees. Re-parent all of it under a floating base_link body.
wb = root.find("worldbody")
base_geoms = list(wb.findall("geom"))          # base_link + imu link geoms
leg_bodies = [b for b in wb.findall("body")
              if b.get("name") in ("hip_yaw_l_link", "hip_yaw_r_link")]

base = ET.Element("body", {"name": "base_link", "pos": f"0 0 {SPAWN_Z}"})
ET.SubElement(base, "freejoint", {"name": "floating_base"})

# base_link inertial straight from the URDF (the weld drops it from the import).
urdf_tree = ET.parse(URDF)
urdf_root = urdf_tree.getroot()
blink = next(l for l in urdf_root.findall("link") if l.get("name") == "base_link")
bi = blink.find("inertial")
bpos = bi.find("origin").get("xyz")
mass = float(bi.find("mass").get("value"))
inr = bi.find("inertia")
ixx, iyy, izz = (float(inr.get(k)) for k in ("ixx", "iyy", "izz"))
ixy, ixz, iyz = (float(inr.get(k)) for k in ("ixy", "ixz", "iyz"))
# MuJoCo fullinertia order: ixx iyy izz ixy ixz iyz
ET.SubElement(base, "inertial", {
    "pos": bpos,
    "mass": f"{mass:g}",
    "fullinertia": f"{ixx:g} {iyy:g} {izz:g} {ixy:g} {ixz:g} {iyz:g}",
})

# IMU mount: read pose from the URDF "imu" fixed joint (the "imu" link is rigidly
# fixed to base_link). The gyro/accelerometer (declared in world.xml) report in this
# site frame, so it must match the IMU mount used in training.
imu_joint = next(j for j in urdf_root.findall("joint") if j.get("name") == "imu")
imu_org = imu_joint.find("origin")
imu_pos = imu_org.get("xyz", "0 0 0") if imu_org is not None else "0 0 0"
imu_rpy = [float(v) for v in
           (imu_org.get("rpy", "0 0 0") if imu_org is not None else "0 0 0").split()]
ET.SubElement(base, "site", {
    "name": "imu", "pos": imu_pos, "quat": rpy_to_quat(*imu_rpy), "size": "0.01",
})

for g in base_geoms:
    wb.remove(g)
    base.append(g)
for b in leg_bodies:
    wb.remove(b)
    base.append(b)

# --- 3. per-joint armature ---
# every <joint> under base is one of the 10 leg hinges (the free base is a separate
# <freejoint> tag); an unknown name means the URDF changed -> KeyError on purpose.
for jnt in base.iter("joint"):
    jnt.set("armature", f"{armature_for(jnt.get('name')):g}")

# --- 4. collision filtering: self-collision OFF, robot<->floor ON ---
# The compile marks visual geoms (group 1) contype=0 conaffinity=0 already. Give the
# collision geoms (no contype set) contype=1 conaffinity=0: they collide with the
# scene.xml floor (conaffinity=1) but, sharing conaffinity=0, never with each other.
for g in base.iter("geom"):
    if g.get("contype") is None:
        g.set("contype", "1")
        g.set("conaffinity", "0")

# --- 5. emit a <mujocoinclude> body-tree ---
out_root = ET.Element("mujocoinclude")
out_root.append(base)
out_tree = ET.ElementTree(out_root)
ET.indent(out_tree, space="    ")
out_tree.write(OUT, encoding="unicode", xml_declaration=False)
print("wrote", OUT)
