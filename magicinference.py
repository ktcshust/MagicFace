import argparse
import os
import gc

import numpy as np
import torch
from torch import autocast
import torchvision.transforms as transforms
from PIL import Image
from diffusers import AutoencoderKL, UniPCMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from rembg import remove

from mgface.pipelines_mgface.pipeline_mgface import MgPipeline as MgPipelineInference
from mgface.pipelines_mgface.unet_ID_2d_condition import UNetID2DConditionModel
from mgface.pipelines_mgface.unet_deno_2d_condition import UNetDeno2DConditionModel

# AU mapping
dict_AU_index = {
    'AU1': 0, 'AU2': 1, 'AU4': 2, 'AU5': 3,
    'AU6': 4, 'AU9': 5, 'AU12': 6, 'AU15': 7,
    'AU17': 8, 'AU20': 9, 'AU25': 10, 'AU26': 11
}

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Optimized MagicFace inference with mixed precision and memory-saving techniques"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='sd-legacy/stable-diffusion-v1-5',
        help="Path or HF identifier of pretrained SD model",
    )
    parser.add_argument(
        "--revision", type=str, default=None,
        help="Revision of pretrained model"
    )
    parser.add_argument(
        "--variant", type=str, default=None,
        help="Variant (e.g. fp16)"
    )
    parser.add_argument(
        "--seed", type=int, default=424,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--inference_steps", type=int, default=40,
        help="Number of denoising steps (reduce to save memory)"
    )
    parser.add_argument(
        "--denoising_unet_path", type=str,
        default='mengtingwei/magicface',
        help="Path to denoising UNet"
    )
    parser.add_argument(
        "--ID_unet_path", type=str,
        default='mengtingwei/magicface',
        help="Path to ID UNet"
    )
    parser.add_argument(
        "--au_test", type=str, default='',
        help="One or multiple AUs to test, e.g. 'AU12' or 'AU1+AU12'"
    )
    parser.add_argument(
        "--AU_variation", type=str, default='',
        help="Variation values, e.g. '10' or '5+8'"
    )
    parser.add_argument(
        "--img_path", type=str, required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--saved_path", type=str, default='edited_images',
        help="Directory to save edited outputs"
    )
    return parser.parse_args(input_args)


def make_data(img_path):
    """
    Load source and generate background-only tensor from input image.
    """
    transform = transforms.ToTensor()
    img = Image.open(img_path).convert("RGBA")
    source = transform(img)
    fg = remove(img)
    _, _, _, alpha = fg.split()
    inv_alpha = alpha.point(lambda v: 255 - v)
    bg = img.copy()
    bg.putalpha(inv_alpha)
    bg_tensor = transform(bg)
    return source, bg_tensor


def tokenize_captions(tokenizer, captions):
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return inputs.input_ids


def main(args):
    # Device & dtype
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weight_dtype = torch.float16

    # Set seed
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Load VAE and CLIP models
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True
    ).to(device)
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True
    ).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )

    # Load custom UNets
    unet_ID = UNetID2DConditionModel.from_pretrained(
        args.ID_unet_path,
        subfolder='ID_enc',
        torch_dtype=weight_dtype,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True
    ).to(device)
    unet_deno = UNetDeno2DConditionModel.from_pretrained(
        args.denoising_unet_path,
        subfolder='denoising_unet',
        torch_dtype=weight_dtype,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True
    ).to(device)

    # Freeze weights
    for module in [vae, text_encoder, unet_ID, unet_deno]:
        module.requires_grad_(False)

    # Build pipeline
    pipeline = MgPipelineInference.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet_ID=unet_ID,
        unet_deno=unet_deno,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True
    ).to(device)

    # Replace scheduler if needed
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

    # Memory optimizations
    pipeline.enable_attention_slicing()
    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    try:
        pipeline.enable_sequential_cpu_offload()
    except Exception:
        pass
    pipeline.set_progress_bar_config(disable=True)

    # Prepare data
    source, bg = make_data(args.img_path)
    source = source.unsqueeze(0).to(device, dtype=weight_dtype)
    bg = bg.unsqueeze(0).to(device, dtype=weight_dtype)

    # Text embeddings
    prompt = "A close up of a person."
    prompt_ids = tokenize_captions(tokenizer, [prompt]).to(device)
    with torch.no_grad():
        prompt_embeds = text_encoder(prompt_ids)[0]

    # AU vector
    au_prompt = np.zeros((12,), dtype=np.float32)
    if '+' in args.au_test:
        aus = args.au_test.split('+')
        vars = args.AU_variation.split('+')
        for au, var in zip(aus, vars):
            au_prompt[dict_AU_index[au]] = float(var)
    elif args.au_test:
        au_prompt[dict_AU_index[args.au_test]] = float(args.AU_variation)
    tor_au = torch.from_numpy(au_prompt).unsqueeze(0).to(device)

    # Inference
    os.makedirs(args.saved_path, exist_ok=True)
    img_name = os.path.basename(args.img_path)
    with torch.no_grad():
        with autocast(device_type=device, dtype=torch.float16):
            sample = pipeline(
                prompt_embeds=prompt_embeds,
                source=source,
                bg=bg,
                au=tor_au,
                num_inference_steps=args.inference_steps,
                generator=generator
            ).images[0]

    out_path = os.path.join(args.saved_path, img_name)
    sample.save(out_path)
    print(f"Saved edited image to {out_path}")

    # Free memory
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    args = parse_args()
    main(args)
