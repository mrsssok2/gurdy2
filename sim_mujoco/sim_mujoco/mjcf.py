"""Generate Gurdy's MuJoCo model (MJCF) from robot.yaml via gurdy_core — same robot as the URDF.

Matches the URDF/Gazebo setup:
  * identical link masses/inertias (explicit <inertial>, no armature), joint axes, limits,
    joint damping + Coulomb friction (frictionloss), effort limits
  * only the feet and the body box collide; no self-collision (Gazebo default)
  * PD actuators: tau = kp*(ctrl - q) - kd*qdot, clipped to ±effort, evaluated every physics step
    (timestep = 1/pd_rate_hz) with the Euler integrator (explicit PD, implicit joint damping)

    python3 -m sim_mujoco.mjcf > gurdy.xml           # write the model
"""
from __future__ import annotations

import sys

import numpy as np

from gurdy_core.kinematics import leg_ik, nominal_foot
from gurdy_core.robot import JOINT_TYPES, LEG_NAMES, Robot, load_robot

_AXIS = {"hip_yaw": "0 0 1", "hip_pitch": "0 -1 0", "knee": "0 1 0"}


def _v(a) -> str:
    return " ".join(f"{float(x):.9g}" for x in a)


def build_mjcf(robot: Robot | None = None, fixed_base: bool = False, base_z: float | None = None,
               hfield: dict | None = None) -> str:
    """hfield (step 2.8 terrain): {"n": grid points per side, "half": half size [m],
    "elevation": height range [m], "z_offset": world z of data value 0}. Adds a heightfield geom
    "terrain" (same contact class as the floor) next to the floor plane; the env fills
    model.hfield_data at runtime and switches the two on/off via contype."""
    r = robot or load_robot()
    l1, l2, l3 = r.lengths
    jc, c, sim = r.joint_common, r.control, r.sim
    effort = jc["effort"]
    foot_r = r.foot_radius
    mu = r.foot_friction
    if base_z is None:
        base_z = 0.3 if fixed_base else r.stance["body_height"] + 0.005

    def seg_inertial(seg):
        s = r.segments[seg]
        return f'<inertial pos="{_v(s.com)}" mass="{s.mass}" diaginertia="{_v(r.link_inertia(seg))}"/>'

    def joint(leg, t):
        lo, hi = r.joint_limits[t]
        return f'<joint name="{leg}_{t}" axis="{_AXIS[t]}" range="{lo} {hi}"/>'

    legs = []
    for leg in LEG_NAMES:
        m = r.mounts[leg]
        fem, tib, cox = r.segments["femur"], r.segments["tibia"], r.segments["coxa"]
        legs.append(f"""
      <body name="{leg}_coxa" pos="{m.xy[0]} {m.xy[1]} 0" euler="0 0 {m.yaw}">
        {joint(leg, "hip_yaw")}
        {seg_inertial("coxa")}
        <geom class="visual" type="box" pos="{l1/2} 0 0" size="{l1/2} {cox.size_yz[0]/2} {cox.size_yz[1]/2}" rgba="0.15 0.35 0.75 1"/>
        <body name="{leg}_femur" pos="{l1} 0 0">
          {joint(leg, "hip_pitch")}
          {seg_inertial("femur")}
          <geom class="visual" type="box" pos="{l2/2} 0 0" size="{l2/2} {fem.size_yz[0]/2} {fem.size_yz[1]/2}" rgba="0.15 0.35 0.75 1"/>
          <body name="{leg}_tibia" pos="{l2} 0 0">
            {joint(leg, "knee")}
            {seg_inertial("tibia")}
            <geom class="visual" type="cylinder" fromto="0 0 0 {l3} 0 0" size="{tib.radius}" rgba="0.95 0.55 0.1 1"/>
            <geom name="{leg}_foot" class="collision" type="sphere" pos="{l3} 0 0" size="{foot_r}" rgba="0.05 0.05 0.05 1"/>
            <site name="{leg}_foot" pos="{l3} 0 0"/>
          </body>
        </body>
      </body>""")

    actuators = "\n".join(
        f'    <general name="{n}" joint="{n}" gainprm="{c["kp"]}" biastype="affine" '
        f'biasprm="0 {-c["kp"]} {-c["kd"]}" forcerange="{-effort} {effort}"/>'
        for n in r.joint_names)
    bx, by, bz = (s / 2 for s in r.body_size)
    cam = r.sensors["depth_camera"]
    p = cam["pitch_down"]
    # MuJoCo cameras look along -z_cam with x_cam right, y_cam up: right = -y_body,
    # up = (sin p, 0, cos p) gives a view direction pitched down by p (same as the URDF camera_link).
    cam_xyaxes = f"0 -1 0 {np.sin(p):.9g} 0 {np.cos(p):.9g}"
    fovy_deg = np.degrees(2 * np.arctan(np.tan(cam["hfov"] / 2) * cam["height"] / cam["width"]))
    q0 = leg_ik(r, nominal_foot(r))
    home_q = " ".join(_v(q0) for _ in LEG_NAMES)
    free_home = "" if fixed_base else f"0 0 {base_z} 1 0 0 0 "
    hf_asset = hf_geom = ""
    if hfield:
        n, half = int(hfield["n"]), float(hfield["half"])
        hf_asset = (f'\n  <asset><hfield name="terrain" nrow="{n}" ncol="{n}" '
                    f'size="{half} {half} {hfield["elevation"]} 0.1"/></asset>')
        hf_geom = (f'\n    <geom name="terrain" type="hfield" hfield="terrain" pos="0 0 {hfield["z_offset"]}" '
                   f'contype="1" conaffinity="0" friction="1.0 0.005 0.0001" rgba="0.55 0.6 0.55 1"/>')
    return f"""<mujoco model="{r.name}">
  <!-- GENERATED from robot.yaml by sim_mujoco/mjcf.py — do not edit by hand. -->
  <compiler angle="radian" autolimits="true" inertiafromgeom="false"/>
  <option timestep="{sim['mujoco_timestep']}" integrator="{sim['mujoco_integrator']}" gravity="0 0 {-sim['gravity']}"/>
  <default>
    <joint damping="{jc['damping']}" frictionloss="{jc['friction']}" armature="0"/>
    <geom contype="0" conaffinity="0"/>
    <default class="visual"><geom group="1" mass="0"/></default>
    <!-- robot geoms only touch the floor (contype 0 / conaffinity 1 vs floor 1 / 0) -->
    <default class="collision"><geom group="0" contype="0" conaffinity="1" friction="{mu} 0.005 0.0001" mass="0"/></default>
  </default>
  <visual><global offwidth="{max(640, cam['width'])}" offheight="{max(480, cam['height'])}"/></visual>{hf_asset}
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="5 5 0.05" contype="1" conaffinity="0" friction="1.0 0.005 0.0001" rgba="0.7 0.7 0.7 1"/>{hf_geom}
    <body name="base_link" pos="0 0 {base_z}">
      {'' if fixed_base else '<freejoint name="root"/>'}
      <inertial pos="0 0 0" mass="{r.body_mass}" diaginertia="{_v(r.body_inertia())}"/>
      <geom name="body" class="collision" type="box" size="{bx} {by} {bz}" rgba="0.25 0.27 0.3 1"/>
      <site name="imu" pos="{_v(r.sensors['imu']['pos'])}"/>
      <camera name="depth_camera" pos="{_v(cam['pos'])}" xyaxes="{cam_xyaxes}" fovy="{fovy_deg:.6g}"/>{''.join(legs)}
    </body>
  </worldbody>
  <actuator>
{actuators}
  </actuator>
  <sensor>
    <gyro name="imu_gyro" site="imu" noise="{r.sensors['imu']['gyro_noise']}"/>
    <accelerometer name="imu_accel" site="imu" noise="{r.sensors['imu']['accel_noise']}"/>
    <framequat name="imu_quat" objtype="site" objname="imu"/>
  </sensor>
  <keyframe>
    <key name="stance" qpos="{free_home}{home_q}" ctrl="{home_q}"/>
  </keyframe>
</mujoco>
"""


if __name__ == "__main__":
    sys.stdout.write(build_mjcf(fixed_base="--fixed" in sys.argv))
