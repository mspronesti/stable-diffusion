from typing import Any
from typing_extensions import Self
import torch.hub as hub


class PreTrainedModel:
    @classmethod
    def from_url(cls, weights_url: str, **kwargs: Any) -> Self:
        model = cls(**kwargs)

        state_dict = hub.load_state_dict_from_url(weights_url, check_hash=True)
        model.load_state_dict(state_dict)
        return model
