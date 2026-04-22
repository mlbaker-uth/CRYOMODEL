"""Sequence-conservation mapping from MSA columns to PDB residues."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import gemmi

from .mutate.sequence import assert_homomultimer_same_sequence, extract_chain_polymer

AA20 = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class ConservationResult:
    rows: List[Dict[str, object]]
    out_csv: Path
    out_json: Optional[Path]
    out_pdb: Optional[Path]


def _read_fasta_alignment(path: Path) -> List[Tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    records: List[Tuple[str, str]] = []
    cur_id: Optional[str] = None
    cur: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur_id is not None:
                records.append((cur_id, "".join(cur).upper()))
            cur_id = line[1:].split()[0] or "seq"
            cur = []
        else:
            cur.append(line.replace(" ", ""))
    if cur_id is not None:
        records.append((cur_id, "".join(cur).upper()))
    if len(records) < 2:
        raise ValueError("Alignment FASTA must contain at least 2 sequences.")
    lengths = {len(seq) for _, seq in records}
    if len(lengths) != 1:
        raise ValueError(f"All alignment sequences must have equal length, got lengths={sorted(lengths)}")
    return records


def _residue_positions_for_alignment(ref_aln: str, residues: Sequence[gemmi.Residue], pdb_seq: str) -> List[Tuple[int, int]]:
    """
    Return ``[(aln_col, residue_idx)]`` for columns that map to modeled residues.

    The reference row may be **longer** than ``pdb_seq`` (e.g. full UniProt vs a model that
    omits N/C termini): non-gap reference letters that do not match the next ``pdb_seq``
    residue are **skipped** (same greedy rule as ``pdb-mutate --alignment-fasta``).
    """
    out: List[Tuple[int, int]] = []
    r_i = 0
    for col, ch in enumerate(ref_aln):
        if r_i >= len(residues):
            break
        if ch == "-":
            continue
        if ch not in AA20:
            raise ValueError(f"Invalid amino acid in reference alignment at column {col}: {ch!r}")
        if pdb_seq[r_i] != ch:
            continue
        out.append((col, r_i))
        r_i += 1
    if r_i != len(residues):
        raise ValueError(
            f"Reference alignment matched {r_i} residues to the PDB chain, but the chain has "
            f"{len(residues)} residues. Check that the first FASTA sequence is the correct "
            "reference for this structure."
        )
    return out


def _penalty(ref: str, aa: str) -> float:
    if aa == ref:
        return 0.0
    grp = {
        "A": "hydrophobic",
        "V": "hydrophobic",
        "I": "hydrophobic",
        "L": "hydrophobic",
        "M": "hydrophobic",
        "F": "aromatic",
        "W": "aromatic",
        "Y": "aromatic",
        "S": "polar",
        "T": "polar",
        "N": "polar",
        "Q": "polar",
        "K": "positive",
        "R": "positive",
        "H": "positive",
        "D": "negative",
        "E": "negative",
        "C": "special",
        "G": "special",
        "P": "special",
    }
    size = {
        "G": 0,
        "A": 0,
        "S": 0,
        "C": 0,
        "P": 0,
        "T": 1,
        "D": 1,
        "N": 1,
        "V": 1,
        "E": 1,
        "Q": 1,
        "H": 1,
        "I": 1,
        "L": 1,
        "M": 2,
        "K": 2,
        "R": 2,
        "F": 2,
        "Y": 2,
        "W": 3,
    }
    p = 0.5 if grp.get(ref) == grp.get(aa) else 2.0
    if abs(size.get(ref, 1) - size.get(aa, 1)) >= 2:
        p += 1.0
    if (grp.get(ref) == "aromatic") ^ (grp.get(aa) == "aromatic"):
        p += 0.5
    return p


def _entropy(counts: Dict[str, int]) -> float:
    n = sum(counts.values())
    if n <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(p)
    return h


def _parse_chain_list(chains: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(chains, str):
        out = [x.strip() for x in chains.split(",") if x.strip()]
    else:
        out = [str(x).strip() for x in chains if str(x).strip()]
    if not out:
        raise ValueError("No chains specified.")
    return out


def build_conservation_rows(
    pdb_path: Path,
    chains: Union[str, Sequence[str]],
    alignment_fasta: Path,
    *,
    include_reference_in_stats: bool = False,
) -> Tuple[List[Dict[str, object]], List[str], str, gemmi.Structure]:
    """
    Load structure, compute per-residue conservation for all chains.

    Returns ``(rows, chain_list, reference_sequence_id, structure)``.
    """
    st = gemmi.read_structure(str(pdb_path))
    chain_list = _parse_chain_list(chains)
    if len(chain_list) > 1:
        assert_homomultimer_same_sequence(st, chain_list)

    ref_chain = chain_list[0]
    pdb_seq, ref_residues = extract_chain_polymer(st, ref_chain)
    aln_records = _read_fasta_alignment(alignment_fasta)
    ref_id, ref_aln = aln_records[0]
    mapping = _residue_positions_for_alignment(ref_aln, ref_residues, pdb_seq)

    stat_records = aln_records if include_reference_in_stats else aln_records[1:]
    if not stat_records:
        stat_records = aln_records
    n_stat = len(stat_records)

    rows: List[Dict[str, object]] = []
    for chain_id in chain_list:
        _, residues = extract_chain_polymer(st, chain_id)
        for col, r_idx in mapping:
            res = residues[r_idx]
            ref = pdb_seq[r_idx]
            letters = [seq[col] for _, seq in stat_records]
            n_gap = sum(1 for ch in letters if ch == "-")
            aa_letters = [ch for ch in letters if ch in AA20]
            counts: Dict[str, int] = {}
            for ch in aa_letters:
                counts[ch] = counts.get(ch, 0) + 1
            n_non_gap = len(aa_letters)
            n_types = len(counts)
            p_nonref = (
                float(sum(1 for ch in aa_letters if ch != ref) / n_non_gap) if n_non_gap > 0 else 0.0
            )
            p_gap = float(n_gap / n_stat) if n_stat > 0 else 0.0
            entropy = _entropy(counts)
            if counts:
                major_aa, major_count = max(counts.items(), key=lambda kv: kv[1])
                p_major = major_count / n_non_gap
            else:
                major_aa, p_major = "", 0.0
            penalties = [_penalty(ref, ch) for ch in aa_letters if ch != ref]
            mean_pen = float(sum(penalties) / max(len(penalties), 1)) if penalties else 0.0
            frac_nonconservative = (
                float(sum(1 for p in penalties if p >= 2.0) / max(len(penalties), 1)) if penalties else 0.0
            )
            rows.append(
                {
                    "chain": chain_id,
                    "seqid": int(res.seqid.num),
                    "icode": res.seqid.icode.strip() if res.seqid.icode else "",
                    "resname": res.name,
                    "ref_aa": ref,
                    "alignment_col": int(col),
                    "n_aa_types": int(n_types),
                    "p_nonref": round(p_nonref, 6),
                    "p_gap": round(p_gap, 6),
                    "entropy": round(entropy, 6),
                    "major_aa": major_aa,
                    "p_major": round(float(p_major), 6),
                    "mean_penalty": round(mean_pen, 6),
                    "frac_nonconservative": round(frac_nonconservative, 6),
                }
            )
    return rows, chain_list, ref_id, st


def compute_conservation(
    pdb_path: Path,
    chains: Union[str, Sequence[str]],
    alignment_fasta: Path,
    *,
    out_csv: Path,
    out_json: Optional[Path] = None,
    out_pdb: Optional[Path] = None,
    bfactor_metric: str = "n_aa_types",
    occupancy_metric: Optional[str] = None,
    include_reference_in_stats: bool = False,
) -> ConservationResult:
    rows, chain_list, ref_id, st = build_conservation_rows(
        pdb_path,
        chains,
        alignment_fasta,
        include_reference_in_stats=include_reference_in_stats,
    )

    out_csv = out_csv.expanduser()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = [
            "chain",
            "seqid",
            "icode",
            "resname",
            "ref_aa",
            "alignment_col",
            "n_aa_types",
            "p_nonref",
            "p_gap",
            "entropy",
            "major_aa",
            "p_major",
            "mean_penalty",
            "frac_nonconservative",
        ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    out_json_path: Optional[Path] = None
    if out_json is not None:
        out_json_path = out_json.expanduser()
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pdb": str(pdb_path),
            "chains": chain_list,
            "alignment_fasta": str(alignment_fasta),
            "reference_sequence_id": ref_id,
            "include_reference_in_stats": include_reference_in_stats,
            "rows": rows,
        }
        out_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out_pdb_path: Optional[Path] = None
    if out_pdb is not None:
        allowed = {
            "n_aa_types",
            "p_nonref",
            "p_gap",
            "entropy",
            "p_major",
            "mean_penalty",
            "frac_nonconservative",
        }
        if bfactor_metric not in allowed:
            raise ValueError(
                f"Unknown bfactor metric {bfactor_metric!r}. Choose one of: {', '.join(sorted(allowed))}"
            )
        if occupancy_metric is not None and occupancy_metric not in allowed:
            raise ValueError(
                f"Unknown occupancy metric {occupancy_metric!r}. Choose one of: {', '.join(sorted(allowed))}"
            )
        chain_set = set(chain_list)
        val_by_chain_seqid = {
            (str(row["chain"]), int(row["seqid"]), str(row["icode"])): float(row[bfactor_metric])
            for row in rows
        }
        occ_by_chain_seqid = (
            {
                (str(row["chain"]), int(row["seqid"]), str(row["icode"])): float(row[occupancy_metric])
                for row in rows
            }
            if occupancy_metric is not None
            else {}
        )
        for model in st:
            for chn in model:
                if chn.name not in chain_set:
                    continue
                for res in chn:
                    key3 = (
                        chn.name,
                        int(res.seqid.num),
                        res.seqid.icode.strip() if res.seqid.icode else "",
                    )
                    if key3 not in val_by_chain_seqid:
                        continue
                    v = val_by_chain_seqid[key3]
                    o = occ_by_chain_seqid.get(key3)
                    for atom in res:
                        atom.b_iso = float(v)
                        if o is not None:
                            atom.occ = float(o)
        out_pdb_path = out_pdb.expanduser()
        out_pdb_path.parent.mkdir(parents=True, exist_ok=True)
        st.write_pdb(str(out_pdb_path))

    return ConservationResult(rows=rows, out_csv=out_csv, out_json=out_json_path, out_pdb=out_pdb_path)

