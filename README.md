# ComfyUI-Krea2-Ostris-Edit

Nodes for running Krea 2 edit LoRAs trained with ai-toolkit (`krea2` arch with
`model_kwargs.edit: true`). 

## Installation

Clone this repo into your ComfyUI `custom_nodes` folder and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ostris/ComfyUI-Krea2-Ostris-Edit.git
```

No extra dependencies are required. The nodes appear under the `ostris/krea2`
category.

## Nodes

### Text Encode Krea 2 Ostris Edit

Inputs: `clip`, `prompt`, optional `vae` and `image1`..`image3`.
Output: `CONDITIONING`.

Encodes the prompt together with the reference images through the Krea 2
Qwen3-VL text encoder, using Krea's conditioning template with
`Picture N:` vision placeholders — the same layout used during training.
When a VAE is connected, each reference image is also VAE-encoded and attached
to the conditioning as reference latents for the reference mode node.

Image sizing matches training: images fed to the Qwen3-VL encoder are
downscaled (never upscaled) to fit 384x384 total pixels; reference latents to
fit 1MP.

Note: the text encoder checkpoint must include the Qwen3-VL vision weights or
the images cannot be encoded.

### Krea 2 Ostris Edit Reference Mode

Inputs: `model`, `enable_reference_mode` (default off), `kv_cache` (default off), `scale_ref_positions` (default off, experimental). Output: `model`.

This node opt-in enables reference-latent conditioning on the Krea 2 model. Each
reference is appended to the image token sequence and conditioned at timestep 0
(the `index_timestep_zero` reference method), and the denoising prediction
covers only the target image tokens.

If `enable_reference_mode` is off, this node is a no-op and returns the model
unchanged.

`kv_cache` caches the reference tokens' attention K/V: they are precomputed in
a single t=0 pass and reused on every denoising step, so the references never
ride along in the per-step sequence (faster, especially at many steps). **The
LoRA must be trained with ai-toolkit's `kv_cache` model kwarg for this to work
properly** — leave it off for normally trained edit LoRAs.

`scale_ref_positions` remaps each reference's RoPE y/x coordinates across the
full target latent grid. It is intended for high-resolution targets where
reference latents are capped near 1MP, to reduce top-left anchoring / reduced-scale copies.

The legacy node (`Krea 2 Ostris Edit Model Patch`) is still included for
backward compatibility with existing workflows.

## Example wiring

```
Load Diffusion Model (krea2) -> Load LoRA -> Krea 2 Ostris Edit Reference Mode -> KSampler
CLIPLoader (krea2) -> Text Encode Krea 2 Ostris Edit (prompt + images + VAE) -> positive
CLIPLoader (krea2) -> Text Encode Krea 2 Ostris Edit (negative prompt)      -> negative
```
