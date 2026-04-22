from pathlib import Path
import csv
import numpy as np
from ..core.types import AssignmentSet

def write_site_features(assigns: AssignmentSet, out_path: Path):
    arr = np.asarray(assigns.centers, dtype=np.float32).reshape(-1,3)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index","x_A","y_A","z_A","top","cluster_id","cluster_size","nearest_ONS","n_ONS_2p8"])
        for a in assigns.assignments:
            i = int(a.index)
            x,y,z = arr[i]
            ex = a.explain or {}
            w.writerow([
                i, f"{x:.3f}", f"{y:.3f}", f"{z:.3f}",
                a.top, int(getattr(a, "cluster_id", -1)),
                int(ex.get("cluster_size", 1)),
                f'{float(ex.get("nearest_ONS", 9.99)):.2f}',
                int(ex.get("n_ONS_2p8", 0)),
            ])
