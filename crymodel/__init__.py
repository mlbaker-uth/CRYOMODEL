"""Backward-compatible import path after package rename; prefer ``cryomodel``."""
from __future__ import annotations

import sys

import cryomodel

sys.modules[__name__] = cryomodel
