from stable_diffusion_pt import pipeline

prompt = (
    "A epic and beautiful rococo werewolf drinking coffee, in a burning coffee shop. ultra-detailed. anime, "
    "pixiv, uhd 8k cryengine, octane render"
)  # @param { type: "string" }
prompts = [prompt]

uncond_prompt = ""  # @param { type: "string" }
uncond_prompts = [uncond_prompt] if uncond_prompt else None

input_images = None

strength = 0.8  # @param { type:"slider", min: 0, max: 1, step: 0.01 }

do_cfg = True  # @param { type: "boolean" }
cfg_scale = 7.5  # @param { type:"slider", min: 1, max: 14, step: 0.5 }
height = 512  # @param { type: "integer" }
width = 512  # @param { type: "integer" }
sampler = "k_euler_ancestral"  # @param ["k_lms", "k_euler", "k_euler_ancestral"]
n_inference_steps = 50  # @param { type: "integer" }

use_seed = False  # @param { type: "boolean" }
if use_seed:
    seed = 42  # @param { type: "integer" }
else:
    seed = None

if __name__ == "__main__":
    img = pipeline.generate(
        prompts=prompts,
        uncond_prompts=uncond_prompts,
        input_images=input_images,
        strength=strength,
        do_cfg=do_cfg,
        cfg_scale=cfg_scale,
        height=height,
        width=width,
        sampler=sampler,
        n_inference_steps=n_inference_steps,
        seed=seed,
        device="cpu",
        idle_device="cpu",
    )[0]

    img.show()
