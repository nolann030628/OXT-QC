import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st


def get_interp1d(t, x):
    gripper_interp = si.interp1d(
        t, x, 
        axis=0, bounds_error=False, 
        fill_value=(x[0], x[-1]))
    return gripper_interp


def get_slerp_interp(t, rot):       # (t, ), (t, 3)
    n_dim = rot.shape[-1]
    if n_dim < 3:
        rot_pad = np.concatenate([rot, np.zeros((rot.shape[0], 3 - rot.shape[-1]))], axis=-1)
    else:
        rot_pad = rot
    rot_interp_pad = st.Slerp(
        t, rot_pad, 
        axis=0, bounds_error=False, 
        fill_value=(rot[0], rot[-1]))
    return rot_interp_pad[..., :n_dim]


class PoseInterpolator:
    def __init__(self, t, x):
        pos = x[:,:3]
        rot = st.Rotation.from_rotvec(x[:,3:])
        self.pos_interp = get_interp1d(t, pos)
        self.rot_interp = st.Slerp(t, rot)
    
    @property
    def x(self):
        return self.pos_interp.x
    
    def __call__(self, t):
        min_t = self.pos_interp.x[0]
        max_t = self.pos_interp.x[-1]
        t = np.clip(t, min_t, max_t)

        pos = self.pos_interp(t)
        rot = self.rot_interp(t)
        rvec = rot.as_rotvec()
        pose = np.concatenate([pos, rvec], axis=-1)
        return pose
    
class LinearInterpolator:
    def __init__(self, t, x):
        self.interp = get_interp1d(t, x)
    
    @property
    def x(self):
        return self.interp.x
    
    def __call__(self, t):
        min_t = self.interp.x[0]
        max_t = self.interp.x[-1]
        t = np.clip(t, min_t, max_t)
        return self.interp(t)