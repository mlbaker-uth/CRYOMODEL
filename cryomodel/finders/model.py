# model.py (baseline now; ML later)
from typing import Dict, Any
CLASSES = ["water","Na+","K+","Mg2+","Ca2+","Cl-","M2+","lipid_head","lipid_tail","sterol","detergent","carbohydrate","ligand","unknown"]

def assign_baseline(feats: Dict[str, Any]) -> Dict[str, float]:
    shell = feats.get("shell_counts", [0,0,0,0,0])
    near = shell[0] + shell[1]           # within ~2.8 Å
    if near >= 6: return {"Mg2+":0.5,"Ca2+":0.2,"water":0.2,"unknown":0.1}
    if near >= 4: return {"water":0.7,"unknown":0.3}
    return {"unknown":1.0}
