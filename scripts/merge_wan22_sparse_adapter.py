#!/usr/bin/env python3
"""Merge a trained SLA proj_l adapter into a Wan2.2 expert checkpoint."""

import argparse
from pathlib import Path

import torch

from rcm.utils.model_utils import load_state_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = load_state_dict(args.base)
    payload = torch.load(args.adapter, map_location="cpu", weights_only=True)
    adapter = payload["adapter"]
    missing = sorted(set(adapter) - set(state))
    if missing:
        raise KeyError(f"Adapter keys absent from base checkpoint: {missing[:5]}")
    for name, value in adapter.items():
        if state[name].shape != value.shape:
            raise ValueError(f"Shape mismatch for {name}: {state[name].shape} != {value.shape}")
        state[name] = value.to(dtype=state[name].dtype)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    print(f"Merged {len(adapter)} tensors into {args.output}")


if __name__ == "__main__":
    main()
