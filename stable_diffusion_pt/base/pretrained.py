from typing import Any

from typing_extensions import Self
import torch.hub as hub
import warnings


class PreTrainedModel:
    @classmethod
    def from_remote_weights(cls, weights_url: str, **kwargs: Any) -> Self:
        model = cls(**kwargs)

        state_dict = hub.load_state_dict_from_url(weights_url, check_hash=True)
        state_dict = make_compatible(state_dict)

        model.load_state_dict(state_dict)
        return model


def make_compatible(state_dict):
    keys = list(state_dict.keys())
    changed = False
    for key in keys:
        if "causal_attention_mask" in key:
            del state_dict[key]
            changed = True
        elif "_proj_weight" in key:
            new_key = key.replace("_proj_weight", "_proj.weight")
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
            changed = True
        elif "_proj_bias" in key:
            new_key = key.replace("_proj_bias", "_proj.bias")
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
            changed = True

    if changed:
        warnings.warn(
            (
                "Given checkpoint data were modified dynamically by make_compatible"
                " function on model_loader.py. Maybe this happened because you're"
                " running newer codes with older checkpoint files. This behavior"
                " (modify old checkpoints and notify rather than throw an error)"
                " will be removed soon, so please download latest checkpoints file."
            )
        )

    return state_dict
