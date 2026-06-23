"""Small JSONL helpers.

Provenance: bio-sfm-trust-audit (experiments/trust_cue_attribution/io_utils.py).
"""

from __future__ import annotations

import json
import os
from typing import Iterable


def read_jsonl(path: str) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(records: Iterable[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        for rec in records:
            handle.write(json.dumps(rec, sort_keys=True) + "\n")
