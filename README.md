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

(TODO: continue)

## Usage
(TODO: explain)

## Example output

| textual prompt | output |
| ---------- | ----- |
| "Italy with cyberpunk style" | ![image](https://user-images.githubusercontent.com/44113430/199224329-84843049-81e8-4e76-b40d-44d4a10e8e3b.png) |
| "Galileo in Disney style" | ![image](https://user-images.githubusercontent.com/44113430/199230127-18179b87-5e7e-4044-b9c2-268c4af01dd7.png) |
| "DSLR photograph of an astronut riding a horse" | ![image](https://user-images.githubusercontent.com/44113430/199240615-b6b354c0-e5fb-439a-8068-f60accbf3963.png)|
| "Realistic Gandalf playing the piano" | ![image](https://user-images.githubusercontent.com/44113430/199231825-bf608087-8eb7-4114-8703-baf220021001.png) |
| "Eren from Attack on Titan drinking wine" | ![image](https://user-images.githubusercontent.com/44113430/199232713-9c74b863-cdb2-4fd0-a808-c92f76bd8190.png) |


---


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
