from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass
class DomainRow:
    enabled: bool = True
    domain: str = "Domain1"
    chain: str = "A"
    start: int = 1
    end: int = 1
    color: str | None = None


class DomainModel:
    def __init__(self, domains: Dict[str, Dict[str, List[Tuple[int, int]]]] | None = None):
        self.domains: Dict[str, Dict[str, List[Tuple[int, int]]]] = OrderedDict()
        if domains:
            self.domains = self._normalize_domains(domains)

    @staticmethod
    def _parse_range(text: str) -> Tuple[int, int]:
        text = str(text).strip()
        if '-' not in text:
            v = int(text)
            return v, v
        a, b = text.split('-', 1)
        start, end = int(a.strip()), int(b.strip())
        if end < start:
            start, end = end, start
        return start, end

    @classmethod
    def _normalize_domains(cls, domains) -> Dict[str, Dict[str, List[Tuple[int, int]]]]:
        out: Dict[str, Dict[str, List[Tuple[int, int]]]] = OrderedDict()
        for domain_name, chain_map in domains.items():
            if not chain_map:
                continue
            dname = str(domain_name)
            out[dname] = OrderedDict()
            for chain_id, ranges in chain_map.items():
                cid = str(chain_id)
                if isinstance(ranges, str):
                    ranges = [ranges]
                norm_ranges = []
                for r in ranges:
                    norm_ranges.append(cls._parse_range(r))
                norm_ranges.sort(key=lambda x: (x[0], x[1]))
                out[dname][cid] = norm_ranges
        return out

    @classmethod
    def from_json_file(cls, path: str | Path):
        data = json.loads(Path(path).read_text())
        return cls(data)

    @classmethod
    def from_txt_file(cls, path: str | Path):
        domains: Dict[str, Dict[str, List[Tuple[int, int]]]] = OrderedDict()
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            toks = line.split()
            if len(toks) < 3:
                raise ValueError(f"Could not parse domain TXT line: {line}")
            chain, domain = toks[0], toks[1]
            ranges = toks[2:]
            domains.setdefault(domain, OrderedDict()).setdefault(chain, [])
            for r in ranges:
                domains[domain][chain].append(cls._parse_range(r))
        return cls(domains)

    @classmethod
    def from_file(cls, path: str | Path):
        path = Path(path)
        if path.suffix.lower() == '.json':
            return cls.from_json_file(path)
        return cls.from_txt_file(path)

    @classmethod
    def from_rows(cls, rows: Iterable[dict | DomainRow]):
        domains: Dict[str, Dict[str, List[Tuple[int, int]]]] = OrderedDict()
        for row in rows:
            if isinstance(row, DomainRow):
                row = row.__dict__
            if not row.get('enabled', True):
                continue
            domain = str(row['domain']).strip()
            chain = str(row['chain']).strip()
            start = int(row['start'])
            end = int(row['end'])
            if end < start:
                start, end = end, start
            domains.setdefault(domain, OrderedDict()).setdefault(chain, []).append((start, end))
        return cls(domains)

    def to_rows(self) -> List[DomainRow]:
        rows: List[DomainRow] = []
        for domain, chain_map in self.domains.items():
            for chain, ranges in chain_map.items():
                for start, end in ranges:
                    rows.append(DomainRow(True, domain, chain, start, end, None))
        return rows

    def domain_names(self) -> List[str]:
        return list(self.domains.keys())

    def to_json_dict(self) -> Dict[str, Dict[str, List[str]]]:
        out: Dict[str, Dict[str, List[str]]] = OrderedDict()
        for domain, chain_map in self.domains.items():
            out[domain] = OrderedDict()
            for chain, ranges in chain_map.items():
                out[domain][chain] = [f"{a}-{b}" if a != b else str(a) for a, b in ranges]
        return out

    def write_json(self, path: str | Path):
        Path(path).write_text(json.dumps(self.to_json_dict(), indent=2) + "\n")

    def write_txt(self, path: str | Path):
        lines = []
        for domain, chain_map in self.domains.items():
            for chain, ranges in chain_map.items():
                for a, b in ranges:
                    token = f"{a}-{b}" if a != b else str(a)
                    lines.append(f"{chain} {domain} {token}")
        Path(path).write_text("\n".join(lines) + ("\n" if lines else ""))

    def rename_domain(self, old_name: str, new_name: str):
        if old_name not in self.domains:
            return
        new_name = new_name.strip()
        if not new_name:
            raise ValueError("New domain name cannot be blank")
        if new_name == old_name:
            return
        payload = self.domains.pop(old_name)
        if new_name in self.domains:
            for chain, ranges in payload.items():
                self.domains[new_name].setdefault(chain, []).extend(ranges)
        else:
            self.domains[new_name] = payload
        self.domains = self._normalize_domains(self.domains)

    def join_rows(self, row_indices: List[int], new_name: str | None = None):
        rows = self.to_rows()
        if not row_indices:
            return
        picked = [rows[i] for i in sorted(set(row_indices)) if 0 <= i < len(rows)]
        if not picked:
            return
        target_name = (new_name or picked[0].domain).strip()
        for row in picked:
            row.domain = target_name
        self.set_from_rows(rows)

    def split_row(self, row_index: int, split_residue: int):
        rows = self.to_rows()
        if row_index < 0 or row_index >= len(rows):
            raise IndexError("Row index out of range")
        row = rows[row_index]
        if split_residue <= row.start or split_residue > row.end:
            raise ValueError("Split residue must be inside selected row range")
        left = DomainRow(True, row.domain, row.chain, row.start, split_residue - 1, row.color)
        right = DomainRow(True, row.domain, row.chain, split_residue, row.end, row.color)
        rows[row_index:row_index + 1] = [left, right]
        self.set_from_rows(rows)

    def set_from_rows(self, rows: Iterable[dict | DomainRow]):
        self.domains = self.from_rows(rows).domains

    def selection_specs(self) -> Dict[str, List[str]]:
        specs = OrderedDict()
        for domain, chain_map in self.domains.items():
            domain_specs = []
            for chain, ranges in chain_map.items():
                joined = ",".join(f"{a}-{b}" if a != b else str(a) for a, b in ranges)
                domain_specs.append(f"/{chain}:{joined}")
            specs[domain] = domain_specs
        return specs
