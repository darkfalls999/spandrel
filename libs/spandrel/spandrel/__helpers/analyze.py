from __future__ import annotations

import torch

from .model_descriptor import StateDict


def analyze_state_dict(model: torch.nn.Module, state_dict: StateDict) -> None:

    func_name = "Analyze State Dict"

    model_keys = set(model.state_dict().keys())
    loaded_keys = set(state_dict.keys())

    def dump_state_dict_summary(
        state_dict: StateDict, filename: str = "dumped_state_dict.yml"
    ):
        """
        Dumps just the keys and shapes of the state dict
        """
        with open(filename, "w") as f:
            f.write("State Dict Structure:\n\n")

            for key, value in state_dict.items():
                f.write(f"{key}: {tuple(value.shape)}\n")

    print(f"\n{func_name}")
    print("Missing keys (in model but not in state dict):")
    for key in model_keys - loaded_keys:
        print(f"  {key}")

    print("\nUnexpected keys (in state dict but not in model):")
    for key in loaded_keys - model_keys:
        print(f"  {key}")

    print("\nMatching keys:", len(model_keys.intersection(loaded_keys)))
    print("Total model keys:", len(model_keys))
    print("Total state dict keys:", len(loaded_keys))

    dump_state_dict_summary(state_dict)


__all__ = ["analyze_state_dict"]
