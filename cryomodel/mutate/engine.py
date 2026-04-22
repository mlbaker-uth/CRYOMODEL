"""Apply sequence-driven mutations with χ1 rotamer optimization."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gemmi

from ..io.mrc import MapVolume, read_map
from .align import mutations_from_aligned_pairs_fasta, mutations_from_target_fasta, read_fasta
from .chi import chi1_quadruple, mean_sidechain_map_value, pick_best_chi1
from .clash import clash_score_for_residue, self_clash_backbone_sidechain
from .sequence import assert_homomultimer_same_sequence, extract_chain_polymer, one_letter_to_name
from .guide_metrics import delta_guide, guide_for_residue, map_volume_mean_std
from .templates import superpose_and_copy_template


@dataclass
class MutateResult:
    """Result of ``mutate_pdb``; ``map_guide_reference`` is set when a map was provided."""

    structure: gemmi.Structure
    mutations: List[Dict[str, Any]]
    alignment: Optional[Tuple[str, str]] = None
    map_guide_reference: Optional[Dict[str, float]] = None


def _find_residue(st: gemmi.Structure, chain_id: str, seqid: gemmi.SeqId) -> gemmi.Residue:
    for res in st[0][chain_id]:
        if res.seqid == seqid:
            return res
    raise KeyError(f"Residue {seqid} not found on chain {chain_id!r}")


def _replace_residue(st: gemmi.Structure, chain_id: str, new_res: gemmi.Residue) -> None:
    ch = st[0][chain_id]
    for i, res in enumerate(ch):
        if res.seqid == new_res.seqid:
            del ch[i]
            ch.add_residue(new_res, pos=i)
            return
    raise RuntimeError(f"Could not replace residue {new_res.seqid} on chain {chain_id}")


def _seqid_sort_key(s: gemmi.SeqId) -> Tuple[int, str]:
    ic = s.icode.strip() if s.icode else ""
    return (int(s.num), ic)


def mutate_pdb(
    pdb_path: str,
    out_pdb: str,
    chain_ids: Sequence[str],
    *,
    target_fasta: Optional[str] = None,
    alignment_fasta: Optional[str] = None,
    alignment_row_a: int = 0,
    alignment_row_b: int = 1,
    map_path: Optional[str] = None,
    weight_rot: float = 0.15,
    weight_map: float = 0.5,
    density_sigma_mult: float = 1.0,
) -> MutateResult:
    """
    Mutate side chains to match a target sequence.

    Provide **either** ``target_fasta`` (one sequence record) **or**
    ``alignment_fasta`` (two aligned rows from a multi-FASTA, gaps as '-'; default rows 0 and 1).
    """
    st = gemmi.read_structure(str(pdb_path))
    if len(st) == 0:
        raise ValueError("Empty structure.")
    chains = [str(c) for c in chain_ids]
    if not chains:
        raise ValueError("No chains specified.")

    if len(chains) > 1:
        assert_homomultimer_same_sequence(st, chains)

    map_vol: Optional[MapVolume] = None
    map_mu = 0.0
    map_sig = 1.0
    map_ref: Optional[Dict[str, float]] = None
    if map_path:
        map_vol = read_map(map_path)
        wm = weight_map
        map_mu, map_sig = map_volume_mean_std(map_vol)
        map_ref = {
            "map_global_mean": float(map_mu),
            "map_global_std": float(map_sig),
            "density_sigma_mult": float(density_sigma_mult),
        }
    else:
        wm = 0.0

    work = st.clone()
    ref_chain = chains[0]
    pdb_seq, residues = extract_chain_polymer(work, ref_chain)
    aln_pair: Optional[Tuple[str, str]] = None

    if alignment_fasta:
        pairs = mutations_from_aligned_pairs_fasta(
            pdb_seq,
            residues,
            alignment_fasta,
            alignment_row_a=alignment_row_a,
            alignment_row_b=alignment_row_b,
        )
    elif target_fasta:
        fasta = read_fasta(target_fasta)
        if len(fasta) != 1:
            raise ValueError(
                f"--target-fasta must contain exactly one sequence record; found {len(fasta)} in {target_fasta}."
            )
        target_seq = next(iter(fasta.values()))
        pairs, aln_pair = mutations_from_target_fasta(pdb_seq, residues, target_seq)
    else:
        raise ValueError("Provide --target-fasta or --alignment-fasta.")

    plan = sorted(
        [(res.seqid, new_letter) for res, new_letter in pairs],
        key=lambda x: _seqid_sort_key(x[0]),
    )

    mutations_log: List[Dict[str, Any]] = []

    for seqid, new_letter in plan:
        for chain_id in chains:
            cur = _find_residue(work, chain_id, seqid)
            new_name = one_letter_to_name(new_letter)
            ri = gemmi.find_tabulated_residue(cur.name)
            old_letter = ri.one_letter_code if ri and ri.found else "?"

            placed = superpose_and_copy_template(new_name, cur)
            placed.seqid = cur.seqid
            if cur.label_seq is not None:
                placed.label_seq = cur.label_seq

            quad = chi1_quadruple(new_name, placed)

            def clash_fn(trial: gemmi.Residue) -> float:
                return clash_score_for_residue(work, chain_id, trial) + self_clash_backbone_sidechain(
                    trial
                )

            def map_fn(trial: gemmi.Residue) -> float:
                if map_vol is None:
                    return 0.0
                return mean_sidechain_map_value(trial, map_vol)

            if quad is not None:
                pick_best_chi1(
                    new_name,
                    placed,
                    quad,
                    clash_fn,
                    map_fn,
                    weight_rot=weight_rot,
                    weight_map=wm,
                )

            before_guide = guide_for_residue(
                work, chain_id, cur, map_vol, map_mu, map_sig, density_sigma_mult
            )
            after_guide = guide_for_residue(
                work, chain_id, placed, map_vol, map_mu, map_sig, density_sigma_mult
            )
            dguide = delta_guide(
                before_guide,
                after_guide,
                map_mu=map_mu,
                map_sig=map_sig,
                sigma_mult=density_sigma_mult,
            )

            _replace_residue(work, chain_id, placed)
            mutations_log.append(
                {
                    "chain": chain_id,
                    "seqid": str(placed.seqid),
                    "from": old_letter,
                    "to": new_letter,
                    "resname": new_name,
                    "guide": {
                        "before": before_guide,
                        "after": after_guide,
                        "delta": dguide,
                    },
                }
            )

    out_path = Path(out_pdb).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work.write_pdb(str(out_path))
    return MutateResult(
        structure=work,
        mutations=mutations_log,
        alignment=aln_pair,
        map_guide_reference=map_ref,
    )
