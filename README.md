# Stable Diffusion in PyTorch [NO Token Required]

Implementation of Stable Diffusion in PyTorch, for personal interest and learning purpose.

The weights were ported from the original implementation and made pytorch-compatible. No huggingface token is required for this
implementation and no manual download. Everything has been automated.

## Installation
Create a virtual environment:

```shell
python3 -m venv .venv
```

and activate it

```shell
source .venv/bin/activate
```

now install the requirements

```shell
pip install -r requirements.txt
```


## Usage
(TODO: explain)

## Acknowledgements

The implementation is based on this [repository](https://github.com/kjsman/stable-diffusion-pytorch), which is based on these repositories:
- [divamgupta/stable-diffusion-tensorflow](https://github.com/divamgupta/stable-diffusion-tensorflow)
- [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
- [huggingface/transformers](https://github.com/huggingface/transformers)
- [huggingface/diffusers](https://github.com/huggingface/diffusers)
- [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

### Difference with the original implementation
- the `ClipTokenizer` has been rewritten to exploit LRU Caching
- the weights and vocabulary download process has entirely automated so that no manual action is required from the user
- the code was slightly cleaned and given more structure
- add a progress bar to track the generation process and provided a generation script
- provided a containerized version (TODO)
