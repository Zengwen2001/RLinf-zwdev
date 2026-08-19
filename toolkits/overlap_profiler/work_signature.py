# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Structural signatures used to reject mismatched profiling workloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


def work_signature(value: Any) -> Any:
    """Describe workload structure without reading or hashing tensor contents."""
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), str(value.dtype))
    if isinstance(value, np.ndarray):
        return ("ndarray", tuple(value.shape), str(value.dtype))
    if isinstance(value, Mapping):
        return tuple(
            (str(key), work_signature(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(work_signature(item) for item in value)
    return (type(value).__name__, value if isinstance(value, (bool, int, str)) else None)
