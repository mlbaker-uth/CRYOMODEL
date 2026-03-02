import numpy as np
from scipy.ndimage import binary_dilation
from sklearn.neighbors import KDTree
from ..core.types import MapVolume, ModelAtoms

# ----- basic thresholders -----
def threshold(vol: MapVolume, t: float): return (vol.data >= t).astype('uint8')
def binarize(vol: MapVolume, t: float):  return (vol.data >= t).astype('bool')

# ----- Å <-> voxel helpers -----
def ang_to_vox(xyz_ang: np.ndarray, vol: MapVolume) -> np.ndarray:
    # xyz_ang (N,3) -> zyx vox
    apix = float(vol.apix)
    origin_vox_xyz = np.array(vol.origin[::-1], dtype=np.float32)  # origin stored as (nx,ny,nz)
    xyz_vox = xyz_ang / apix + origin_vox_xyz
    zyx_vox = xyz_vox[:, ::-1]
    return zyx_vox

def vox_to_ang(zyx_vox: np.ndarray, vol: MapVolume) -> np.ndarray:
    apix = float(vol.apix)
    origin_vox_xyz = np.array(vol.origin[::-1], dtype=np.float32)
    xyz_vox = zyx_vox[:, ::-1]
    xyz_ang = (xyz_vox - origin_vox_xyz) * apix
    return xyz_ang

# ----- trilinear sampling -----
def sample_trilinear(data: np.ndarray, pts_zyx: np.ndarray) -> np.ndarray:
    z, y, x = pts_zyx[:,0], pts_zyx[:,1], pts_zyx[:,2]
    z0 = np.floor(z).astype(int); y0 = np.floor(y).astype(int); x0 = np.floor(x).astype(int)
    z1 = z0 + 1; y1 = y0 + 1; x1 = x0 + 1
    z0 = np.clip(z0, 0, data.shape[0]-1); z1 = np.clip(z1, 0, data.shape[0]-1)
    y0 = np.clip(y0, 0, data.shape[1]-1); y1 = np.clip(y1, 0, data.shape[1]-1)
    x0 = np.clip(x0, 0, data.shape[2]-1); x1 = np.clip(x1, 0, data.shape[2]-1)
    dz = z - z0; dy = y - y0; dx = x - x0
    c000 = data[z0, y0, x0]; c001 = data[z0, y0, x1]; c010 = data[z0, y1, x0]; c011 = data[z0, y1, x1]
    c100 = data[z1, y0, x0]; c101 = data[z1, y0, x1]; c110 = data[z1, y1, x0]; c111 = data[z1, y1, x1]
    c00 = c000*(1-dx) + c001*dx
    c01 = c010*(1-dx) + c011*dx
    c10 = c100*(1-dx) + c101*dx
    c11 = c110*(1-dx) + c111*dx
    c0 = c00*(1-dy) + c01*dy
    c1 = c10*(1-dy) + c11*dy
    return c0*(1-dz) + c1*dz

def density_at_points(vol: MapVolume, xyz_ang: np.ndarray) -> np.ndarray:
    pts_zyx = ang_to_vox(xyz_ang, vol)
    return sample_trilinear(vol.data, pts_zyx)

# ----- model mask (dilated spheres around atoms) -----
def mask_by_model(vol: MapVolume, model: ModelAtoms, radius_A: float = 2.0) -> np.ndarray:
    # draw spheres on a coarse grid -> dilate; cheap but effective
    r_vox = max(1, int(np.ceil(radius_A / float(vol.apix))))
    mask = np.zeros(vol.data.shape, dtype=bool)
    pts_zyx = ang_to_vox(model.xyz.astype(np.float32), vol).round().astype(int)
    Z, Y, X = vol.data.shape
    pts_zyx[:,0] = np.clip(pts_zyx[:,0], 0, Z-1)
    pts_zyx[:,1] = np.clip(pts_zyx[:,1], 0, Y-1)
    pts_zyx[:,2] = np.clip(pts_zyx[:,2], 0, X-1)
    mask[pts_zyx[:,0], pts_zyx[:,1], pts_zyx[:,2]] = True
    if r_vox > 0:
        # spherical structure element approx by repeated 3D dilation
        selem = np.zeros((3,3,3), bool); selem[:] = True
        for _ in range(r_vox):
            mask = binary_dilation(mask, structure=selem)
    return mask

def apply_mask(vol: MapVolume, mask: np.ndarray, invert: bool = False) -> MapVolume:
    data = vol.data.copy()
    if invert:
        data[mask] = 0.0
    else:
        data[~mask] = 0.0
    return MapVolume(data=data.astype(np.float32), apix=vol.apix, origin=vol.origin, halfmaps=vol.halfmaps)

# ----- distance gate (Å window to nearest model atom) -----
def within_distance_window(xyz_ang: np.ndarray, model: ModelAtoms, dmin: float, dmax: float) -> np.ndarray:
    tree = KDTree(model.xyz.astype(np.float32))
    d, _ = tree.query(xyz_ang.astype(np.float32), k=1, return_distance=True)
    d = d.reshape(-1)
    keep = (d >= dmin) & (d <= dmax)
    return keep

# ----- density sampling -----
def sample_line_density(vol: MapVolume, p0, p1, n: int = 20) -> float:
    ts = np.linspace(0.0, 1.0, n)
    pts = (1-ts)[:,None]*p0[None,:] + ts[:,None]*p1[None,:]
    pts = np.clip(pts, [0,0,0], np.array(vol.data.shape)-1)
    vals = vol.data[pts[:,0].astype(int), pts[:,1].astype(int), pts[:,2].astype(int)]
    return float(vals.mean())
