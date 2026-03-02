import numpy as np
from ..maps.fft import fourier_gaussian_lowpass, add_colored_noise

ELEMENT_RADIUS = {"H":0.8,"C":1.7,"N":1.55,"O":1.52,"P":1.8,"S":1.8,"MG":0.72,"NA":1.02,"K":1.38,"CA":1.0,"CL":1.75}
ELEMENT_AMP    = {"H":0.2,"C":1.0,"N":1.2,"O":1.4,"P":1.6,"S":1.4,"MG":1.8,"NA":1.6,"K":1.6,"CA":1.8,"CL":1.5}

def render_patch(at_xyz, at_el, center_xyz, boxA=24.0, apix=1.0, target_res=3.0, snr=5.0):
    half = boxA/2.0; grid = int(np.ceil(boxA/apix))
    vol = np.zeros((grid,grid,grid), np.float32)
    rel = at_xyz - center_xyz[None,:]
    mask = np.all(np.abs(rel) <= half+3.0, axis=1)
    rel = rel[mask]; el = at_el[mask]
    origin = np.array([grid/2,grid/2,grid/2], np.float32)
    for p,e in zip(rel, el):
        amp = ELEMENT_AMP.get(str(e).upper(), 1.0)
        r   = ELEMENT_RADIUS.get(str(e).upper(), 1.6)
        pos = origin + (p/apix)[::-1]  # Å→voxels (z,y,x)
        cz,cy,cx = np.round(pos).astype(int)
        for dz in range(-2,3):
            for dy in range(-2,3):
                for dx in range(-2,3):
                    z,y,x = cz+dz, cy+dy, cx+dx
                    if 0<=z<grid and 0<=y<grid and 0<=x<grid:
                        d2 = (dz*dz+dy*dy+dx*dx) * (apix**2)
                        vol[z,y,x] += amp*np.exp(-d2/(2*(r**2)))
    vol = fourier_gaussian_lowpass(vol, apix=apix, resA=target_res)
    return add_colored_noise(vol, snr=snr)
