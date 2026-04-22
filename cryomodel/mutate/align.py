"""Pairwise global alignment and FASTA parsing."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

_AA = set("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path: str | Path) -> Dict[str, str]:
    """Parse FASTA into {header_id: sequence}."""
    text = Path(path).expanduser().read_text()
    records: Dict[str, str] = {}
    cur_id: str | None = None
    cur_lines: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur_id is not None:
                records[cur_id] = "".join(cur_lines).replace(" ", "")
            cur_id = line[1:].split()[0]
            cur_lines = []
        else:
            cur_lines.append(line)
    if cur_id is not None:
        records[cur_id] = "".join(cur_lines).replace(" ", "")
    return records


def sequence_from_fasta_row(path: str | Path, row_index: int) -> str:
    """
    Return one sequence from a multi-record FASTA by **0-based file order** (header order).

    Gaps (``-``) are stripped for use as an **unaligned** target (e.g. one MSA row passed to
    :func:`mutations_from_target_fasta`).
    """
    d = read_fasta(path)
    items = list(d.items())
    n = len(items)
    if n < 1:
        raise ValueError(f"Expected at least 1 FASTA record in {path}, found 0")
    if row_index < 0 or row_index >= n:
        raise ValueError(f"Alignment FASTA has {n} record(s); row index {row_index} is out of range.")
    raw = items[row_index][1]
    return _gap_stripped_upper(raw)


def read_aligned_pair_from_fasta(
    path: str | Path, *, row_a: int = 0, row_b: int = 1
) -> Tuple[str, str]:
    """
    Read two sequences from a multi-record FASTA by **0-based record index** (file order).

    Template vs target is decided later by :func:`select_aligned_template_and_target_rows`.
    """
    d = read_fasta(path)
    items = list(d.items())
    n = len(items)
    if n < 2:
        raise ValueError(f"Expected at least 2 FASTA records in {path}, found {n}")
    if row_a < 0 or row_b < 0 or row_a >= n or row_b >= n:
        raise ValueError(
            f"Alignment FASTA has {n} record(s); row indices {row_a}, {row_b} are out of range."
        )
    if row_a == row_b:
        raise ValueError("Alignment row indices must differ (two distinct sequences).")
    return items[row_a][1], items[row_b][1]


def read_two_sequence_fasta(path: str | Path) -> Tuple[str, str]:
    """Read the first two FASTA records; same as ``read_aligned_pair_from_fasta(..., 0, 1)``."""
    return read_aligned_pair_from_fasta(path, row_a=0, row_b=1)


def _gap_stripped_upper(s: str) -> str:
    return s.upper().replace("-", "")


def template_alignment_covers_pdb_sequence(
    pdb_seq: str, template_row: str, target_row: str
) -> bool:
    """
    Greedy column walk: ``pdb_seq`` must be found in order along **template** residues.

    Extra template letters (full-length reference vs coordinates that omit termini or loops)
    are skipped until the next letter matches the next modeled residue.
    """
    if len(template_row) != len(target_row):
        return False
    pdb_i = 0
    for ap, at in zip(template_row.upper(), target_row.upper()):
        if pdb_i >= len(pdb_seq):
            break
        if ap == "-" and at == "-":
            continue
        if ap == "-" and at != "-":
            continue
        if ap == "-":
            continue
        if ap != pdb_seq[pdb_i]:
            continue
        pdb_i += 1
    return pdb_i == len(pdb_seq)


def select_aligned_template_and_target_rows(
    pdb_seq: str, raw_row1: str, raw_row2: str
) -> Tuple[str, str]:
    """
    Pick template (structure reference) vs mutate-to target rows.

    The **template** row is the one that contains ``pdb_seq`` as a subsequence in column
    order; the other row is the target sequence.
    """
    r1 = raw_row1.upper()
    r2 = raw_row2.upper()
    ok1 = template_alignment_covers_pdb_sequence(pdb_seq, r1, r2)
    ok2 = template_alignment_covers_pdb_sequence(pdb_seq, r2, r1)
    if ok1 and not ok2:
        return r1, r2
    if ok2 and not ok1:
        return r2, r1
    if ok1 and ok2:
        raise ValueError(
            "Ambiguous alignment: both FASTA rows work as the template vs the PDB chain."
        )
    head = 80
    ps = pdb_seq[:head] + ("..." if len(pdb_seq) > head else "")
    raise ValueError(
        "Could not match the PDB chain to either FASTA row as a subsequence of the template.\n"
        f"  PDB chain ({len(pdb_seq)} aa): {ps}\n"
        "The template row must contain the coordinate sequence in order (extra reference "
        "residues are skipped). Check chain ID and alignment."
    )


def select_aligned_pdb_and_target_rows(
    pdb_seq: str, raw_row1: str, raw_row2: str
) -> Tuple[str, str]:
    """Backward-compatible alias for :func:`select_aligned_template_and_target_rows`."""
    return select_aligned_template_and_target_rows(pdb_seq, raw_row1, raw_row2)


def mutations_from_aligned_template_subsequence(
    pdb_seq: str,
    residues: List,
    template_row: str,
    target_row: str,
) -> List[Tuple[object, str]]:
    """
    Pairwise alignment where the **template** row may be longer than ``pdb_seq``.

    Match ``pdb_seq`` greedily to template letters; emit substitutions where template and
    target both have residues and differ. Target gaps keep the modeled residue.
    """
    if len(template_row) != len(target_row):
        raise ValueError("Aligned template and target strings must have equal length.")
    out: List[Tuple[object, str]] = []
    pdb_i = 0
    for col_idx, (ap, at) in enumerate(zip(template_row.upper(), target_row.upper())):
        if pdb_i >= len(pdb_seq):
            break
        if ap == "-" and at == "-":
            continue
        if ap == "-" and at != "-":
            continue
        if ap == "-":
            continue
        if ap != pdb_seq[pdb_i]:
            continue
        if at == "-" or at == ".":
            pdb_i += 1
            continue
        if len(at) != 1 or at not in _AA:
            raise ValueError(f"Invalid target residue at column {col_idx}: {at!r}")
        if ap != at:
            out.append((residues[pdb_i], at))
        pdb_i += 1
    if pdb_i != len(pdb_seq):
        raise ValueError(
            "Alignment did not cover all modeled residues: "
            f"matched {pdb_i} of {len(pdb_seq)}. "
            "Check that the template row is the reference for this structure."
        )
    return out


def align_global_simple(
    s1: str,
    s2: str,
    *,
    match: int = 1,
    mismatch: int = -1,
    indel: int = -2,
) -> Tuple[str, str]:
    """Needleman–Wunsch global alignment with linear gap penalty."""
    n, m = len(s1), len(s2)
    score = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = indel * i
    for j in range(1, m + 1):
        score[0][j] = indel * j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            ms = match if s1[i - 1] == s2[j - 1] else mismatch
            score[i][j] = max(
                score[i - 1][j - 1] + ms,
                score[i - 1][j] + indel,
                score[i][j - 1] + indel,
            )
    i, j = n, m
    a1: List[str] = []
    a2: List[str] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            ms = match if s1[i - 1] == s2[j - 1] else mismatch
            if score[i][j] == score[i - 1][j - 1] + ms:
                a1.append(s1[i - 1])
                a2.append(s2[j - 1])
                i -= 1
                j -= 1
                continue
        if i > 0 and score[i][j] == score[i - 1][j] + indel:
            a1.append(s1[i - 1])
            a2.append("-")
            i -= 1
            continue
        a1.append("-")
        a2.append(s2[j - 1])
        j -= 1
    return "".join(reversed(a1)), "".join(reversed(a2))


def mutations_from_alignment(
    pdb_seq: str,
    residues: List,
    aligned_pdb: str,
    aligned_target: str,
) -> List[Tuple[object, str]]:
    """
    Map aligned columns to substitution list (residue, target_one_letter).

    Insertions in the target (PDB gap, target residue) are rejected. Columns where the
    target is gapped (PDB residue, target ``'-'``) are skipped: the modeled residue is
    left unchanged.
    """
    if len(aligned_pdb) != len(aligned_target):
        raise ValueError("Aligned strings must have equal length.")
    out: List[Tuple[object, str]] = []
    pdb_i = 0
    for ap, at in zip(aligned_pdb, aligned_target):
        if ap == "-" and at == "-":
            continue
        if ap == "-" and at != "-":
            raise ValueError(
                "Insertion in target relative to PDB is not supported "
                f"(target column {at!r})."
            )
        if ap != "-" and at == "-":
            # Gap in target: keep modeled PDB residue; do not treat as a substitution.
            pdb_i += 1
            continue
        if len(at) != 1 or at.upper() not in _AA:
            raise ValueError(f"Invalid target residue code: {at!r}")
        if ap != pdb_seq[pdb_i]:
            raise RuntimeError(
                f"Internal alignment error: PDB column {ap!r} at index {pdb_i}, "
                f"expected {pdb_seq[pdb_i]!r}."
            )
        if ap != at.upper():
            out.append((residues[pdb_i], at.upper()))
        pdb_i += 1
    if pdb_i != len(pdb_seq):
        raise RuntimeError("Alignment does not span the full PDB sequence.")
    return out


def mutations_from_aligned_pairs_fasta(
    pdb_seq: str,
    residues: List,
    path: str | Path,
    *,
    alignment_row_a: int = 0,
    alignment_row_b: int = 1,
) -> List[Tuple[object, str]]:
    """
    Multi-record FASTA: pick two rows by index (default first two). Equal-length strings:
    aligned **template** (structure reference) and aligned **target** (gaps as '-').

    The template row may **not** equal ``pdb_seq`` when gaps are stripped: full-length
    reference sequences are aligned by skipping **extra** template residues until each
    modeled residue matches in order. **Which row is template** is auto-detected. See
    :func:`mutations_from_aligned_template_subsequence`.
    """
    raw1, raw2 = read_aligned_pair_from_fasta(
        path, row_a=alignment_row_a, row_b=alignment_row_b
    )
    if len(raw1) != len(raw2):
        raise ValueError("The two sequences in the alignment FASTA must have equal length.")
    s1, s2 = select_aligned_template_and_target_rows(pdb_seq, raw1, raw2)
    return mutations_from_aligned_template_subsequence(pdb_seq, residues, s1, s2)


def mutations_from_target_fasta(
    pdb_seq: str, residues: List, target_seq: str
) -> Tuple[List[Tuple[object, str]], Tuple[str, str]]:
    """
    Compare PDB polymer to a target sequence.

    - If the target has the **same length** as the modeled polymer, residues are paired
      in order (trimmed FASTA matching the model only).
    - If the target is **longer** (e.g. full Uniprot), pair by **author residue number**:
      target letter at index ``seqid.num - 1`` for each residue (handles missing loops in
      the coordinates without shifting the rest of the sequence).
    - Otherwise, fall back to global pairwise alignment (may fail if indels are required).
    """
    tgt = target_seq.upper().replace(" ", "").replace("\n", "")
    for c in tgt:
        if c not in _AA:
            raise ValueError(f"Invalid character in target sequence: {c!r}")
    if not residues:
        return [], (pdb_seq, pdb_seq)
    L = len(pdb_seq)
    n = len(tgt)
    hi = max(int(r.seqid.num) for r in residues)
    if n == L:
        seg = tgt
        aln_pair: Tuple[str, str] = (pdb_seq, seg)
    elif n >= hi:
        seg = "".join(tgt[int(r.seqid.num) - 1] for r in residues)
        if len(seg) != L:
            raise RuntimeError("Internal error: target segment length mismatch.")
        aln_pair = (pdb_seq, seg)
    else:
        aln_p, aln_t = align_global_simple(pdb_seq, tgt)
        return mutations_from_alignment(pdb_seq, residues, aln_p, aln_t), (aln_p, aln_t)

    pairs: List[Tuple[object, str]] = []
    for i, res in enumerate(residues):
        if pdb_seq[i] != seg[i]:
            pairs.append((res, seg[i]))
    return pairs, aln_pair
