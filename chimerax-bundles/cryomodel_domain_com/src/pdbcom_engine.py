from __future__ import annotations

import csv
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

from chimerax.core.commands import run as cxrun


class PDBComEngine:
    def __init__(self, executable: str = "cryomodel"):
        self.executable = executable or "cryomodel"

    def ensure_work_dir(self, work_dir: str | None = None) -> Path:
        if work_dir:
            wd = Path(work_dir)
            wd.mkdir(parents=True, exist_ok=True)
            return wd
        return Path(tempfile.mkdtemp(prefix="cryomodel_domain_com_"))

    def export_model(self, session, structure, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Use save command so CryoModel sees a regular PDB file.
        cxrun(session, f'save "{out_path}" #{structure.id_string}')
        return out_path

    def build_pdbdomain_cmd(
        self,
        model_path: str | Path,
        chain: str | None,
        n_domains: int | None,
        out_prefix: str,
        options: Dict | None = None,
    ) -> List[str]:
        opts = dict(options or {})
        cmd = [self.executable, "pdbdomain", "--model", str(model_path), "--out-prefix", str(out_prefix)]
        if chain:
            cmd.extend(["--chain", str(chain)])
        if n_domains is not None:
            cmd.extend(["--n-domains", str(int(n_domains))])
        if "merge_distance" in opts:
            cmd.extend(["--merge-distance", str(opts["merge_distance"])])
        if "seed_size" in opts:
            cmd.extend(["--seed-size", str(int(opts["seed_size"]) )])
        if "min_domain_residues" in opts:
            cmd.extend(["--min-domain-residues", str(int(opts["min_domain_residues"]) )])
        if "prefer_gaps" in opts:
            cmd.append("--prefer-gaps" if opts["prefer_gaps"] else "--no-prefer-gaps")
        if "gap_window" in opts:
            cmd.extend(["--gap-window", str(int(opts["gap_window"]) )])
        if "gaps_only" in opts:
            cmd.append("--gaps-only" if opts["gaps_only"] else "--no-gaps-only")
        if "sse_source" in opts and opts["sse_source"]:
            cmd.extend(["--sse-source", str(opts["sse_source"])])
        if "sse_window" in opts:
            cmd.extend(["--sse-window", str(int(opts["sse_window"]) )])
        if "write_pdb" in opts:
            cmd.append("--write-pdb" if opts["write_pdb"] else "--no-write-pdb")
        return cmd

    def build_pdbcom_cmd(
        self,
        model_path: str | Path,
        domains_json_path: str | Path,
        out_prefix: str,
        options: Dict | None = None,
    ) -> List[str]:
        opts = dict(options or {})
        cmd = [
            self.executable, "pdbcom",
            "--model", str(model_path),
            "--domains", str(domains_json_path),
            "--out-prefix", str(out_prefix),
        ]
        if "mass_weighted" in opts:
            cmd.append("--mass-weighted" if opts["mass_weighted"] else "--no-mass-weighted")
        if "atoms" in opts and opts["atoms"]:
            cmd.extend(["--atoms", str(opts["atoms"])])
        return cmd

    def run_command(self, cmd: List[str], work_dir: str | Path):
        proc = subprocess.run(cmd, cwd=str(work_dir), capture_output=True, text=True)
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "command": cmd,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "work_dir": str(work_dir),
        }

    def run_pdbdomain(self, model_path, chain, n_domains, out_prefix, work_dir=None, options=None):
        wd = self.ensure_work_dir(work_dir)
        cmd = self.build_pdbdomain_cmd(model_path, chain, n_domains, out_prefix, options)
        result = self.run_command(cmd, wd)
        base = wd / out_prefix
        result.update({
            "json_file": str(base.with_suffix('.json')),
            "csv_file": str(base.with_suffix('.csv')),
            "pdb_file": str(base.with_suffix('.pdb')),
        })
        return result

    def run_pdbcom(self, model_path, domains_json_path, out_prefix, work_dir=None, options=None):
        wd = self.ensure_work_dir(work_dir)
        cmd = self.build_pdbcom_cmd(model_path, domains_json_path, out_prefix, options)
        result = self.run_command(cmd, wd)
        base = wd / out_prefix
        result.update({
            "pdb_file": str(base.with_suffix('.pdb')),
            "csv_file": str(base.with_suffix('.csv')),
        })
        return result

    def parse_com_csv(self, csv_path: str | Path):
        rows = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                norm = {str(k).strip().lower(): v for k, v in row.items()}
                rows.append({
                    "domain": norm.get("domain", ""),
                    "x": float(norm.get("x", 0.0) or 0.0),
                    "y": float(norm.get("y", 0.0) or 0.0),
                    "z": float(norm.get("z", 0.0) or 0.0),
                    "num_atoms": int(float(norm.get("num_atoms", 0) or 0)),
                    "mass": float(norm.get("mass", 0.0) or 0.0),
                    "chains": norm.get("chains", ""),
                })
        return rows
