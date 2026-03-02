# crymodel/domains/__init__.py
"""Domain analysis tools."""
from .pdbcom import compute_domain_coms, write_domain_com_pdb

__all__ = [
    "compute_domain_coms",
    "write_domain_com_pdb",
]

