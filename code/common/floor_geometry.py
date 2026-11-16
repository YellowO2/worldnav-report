"""Floor geometry with fixed per-camera constants: a floor pixel -> its ground point, no depth model."""
import json
import math

import numpy as np


def group(name):
    """Camera group of a trajectory name (the robot and, for Go Stanford, real vs simulated)."""
    n = name.split('/')[-1]
    if n.startswith('jackal_'):
        return 'recon'
    if n.startswith('random_mdps_'):
        return 'scand_jackal' if 'Jackal' in n else 'scand_spot'
    if n.startswith('sim'):
        return 'stanford_sim'
    if n.startswith('no'):
        return 'stanford_real'
    return 'huron'


GROUPS = ['scand_spot', 'scand_jackal', 'recon', 'huron', 'stanford_real', 'stanford_sim', 'tartan']
NATIVE_SIZE = {'scand_spot': (320, 240), 'scand_jackal': (320, 240), 'recon': (640, 480), 'huron': (160, 120),
               'stanford_real': (160, 120), 'stanford_sim': (160, 120), 'tartan': (1024, 544)}


def group_of(trajectory, split):
    """Camera group."""
    return 'tartan' if split == 'val_unseen' else group(trajectory)


# Go Stanford real photos run 2 frames behind the recorded motion
PHOTO_SHIFT = {'stanford_real': 2}


def start_photo(frame_list, grp, shift):
    """The (offset, size) of the photo a route is drawn on: photo 0."""
    return frame_list[min(PHOTO_SHIFT.get(grp, 0), len(frame_list) - 1) if shift else 0]


def load_cameras():
    kb = json.load(open('config/fisheye_anycalib.json'))
    da3 = json.load(open('config/cameras_undistorted_anycalib_gtposes.json'))
    def tilt(s):   # the calibration's tilt already has the sign used here
        return da3[s]['tilt_deg']   # eval/fit_floor_camera.py: Go Stanford -4.5 deg vs calibration -4.8
                                    # far floor points too high, onto walls)
    return {
        'scand_spot': dict(lens='pinhole', fx=213.3, fy=212.3, cx=160.0, cy=120.0, height=0.76, tilt=0.0),
        'scand_jackal': dict(lens='pinhole', fx=213.3, fy=212.3, cx=160.0, cy=120.0, height=0.40, tilt=12.5),
        'recon': dict(lens='kb', params=kb['recon']['params'], height=0.46, tilt=tilt('recon')),
        'huron': dict(lens='kb', params=kb['huron']['params'], height=0.26, tilt=tilt('huron')),
        'stanford_real': dict(lens='kb', params=kb['stanford']['params'], height=0.46, tilt=tilt('stanford')),
        'stanford_sim': dict(lens='kb', params=kb['stanford']['params'], height=0.38, tilt=tilt('stanford')),
        # TartanDrive: published rectified MultiSense calibration (tartan_drive README)
        'tartan': dict(lens='pinhole', fx=477.605, fy=477.605, cx=499.5, cy=252.0, tilt=13.0,
                       height=_fitted_height('tartan', 1.5)),
    }


def _fitted_height(grp, default):
    try:
        return float(json.load(open(f'config/floor_fit_{grp}.json'))['height'])
    except (OSError, KeyError, ValueError):
        return default


def ray(u, v, cam):
    """Unit viewing direction (x right, y down, z forward) of pixel (u, v)."""
    if cam['lens'] == 'pinhole':
        d = np.array([(u - cam['cx']) / cam['fx'], (v - cam['cy']) / cam['fy'], 1.0])
        return d / np.linalg.norm(d)
    fx, fy, cx, cy, k1, k2, k3, k4 = cam['params']
    mx, my = (u - cx) / fx, (v - cy) / fy
    rd = math.hypot(mx, my)                        # = theta * (1 + k1 theta^2 + ...): solve for theta (Newton)
    th = rd
    for _ in range(20):
        g = th * (1 + k1 * th ** 2 + k2 * th ** 4 + k3 * th ** 6 + k4 * th ** 8) - rd
        dg = 1 + 3 * k1 * th ** 2 + 5 * k2 * th ** 4 + 7 * k3 * th ** 6 + 9 * k4 * th ** 8
        th -= g / dg
    if rd < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    s = math.sin(th) / rd
    return np.array([mx * s, my * s, math.cos(th)])


def axes(cam):
    """The camera's floor frame in camera coordinates (x right, y down, z forward): down, forward."""
    if cam.get('gravity') is not None:
        down = np.asarray(cam['gravity'], dtype=float)
        down = down / np.linalg.norm(down)
    else:
        t = math.radians(cam['tilt'])
        down = np.array([0.0, math.cos(t), math.sin(t)])
    fwd = np.array([0.0, 0.0, 1.0]) - down[2] * down
    fwd /= np.linalg.norm(fwd)
    return down, fwd, np.cross(down, fwd)


def with_gravity(cam, gravity):
    return dict(cam, gravity=None if gravity is None else [float(v) for v in gravity])


def ground_point(u, v, cam):
    """(forward, right) metres on the floor, or None above the horizon."""
    d = ray(u, v, cam)
    down, fwd, right = axes(cam)
    dd = float(d @ down)
    if dd <= 1e-4:
        return None
    k = cam['height'] / dd
    return (k * float(d @ fwd), k * float(d @ right))


def lift_path(pixels, cam, max_jump=3.0):
    """Pixel dots (robot first) -> floor path."""
    pts = [(0.0, 0.0)]
    for u, v in pixels:
        g = ground_point(u, v, cam)
        if g is not None:
            pts.append(g)
    pts = np.array(pts)
    if len(pts) < 3:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cap = max_jump * float(np.median(seg))
    out = [pts[0]]
    for a, b, s in zip(pts, pts[1:], seg):
        b = out[-1] + (b - a) * min(1.0, cap / max(s, 1e-9))
        out.append(b)
    return np.array(out)


def project(fwd, right, cam):
    """A floor point (forward, right) metres -> its pixel (u, v), or None if behind the camera."""
    down, f, r = axes(cam)
    x, y, z = cam['height'] * down + fwd * f + right * r
    if z <= 1e-3:
        return None
    if cam['lens'] == 'pinhole':
        return cam['fx'] * x / z + cam['cx'], cam['fy'] * y / z + cam['cy']
    fx, fy, cx, cy, k1, k2, k3, k4 = cam['params']
    rr = math.hypot(x, y)
    th = math.atan2(rr, z)
    thd = th * (1 + k1 * th ** 2 + k2 * th ** 4 + k3 * th ** 6 + k4 * th ** 8)
    if rr < 1e-9:
        return cx, cy
    return cx + fx * thd * x / rr, cy + fy * thd * y / rr
