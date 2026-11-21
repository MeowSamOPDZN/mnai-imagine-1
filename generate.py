"""Make pictures with a trained Imagine 1 checkpoint. DDIM sampling.

Usage:
  python generate.py --model out-im1/ckpt-500 --prompt "a photo of a cat"
  python generate.py --model out-im1/ckpt-500 --prompt "a photo of a dog" --n 4 --out cats.png
  python generate.py --model out-im1/ckpt-500 --prompt "a photo of a cat" --steps 100 --cfg 7.5
"""
import argparse
import json
import math
import os

import torch
import torch.nn.functional as F
from PIL import Image

from model import IMAGE_SIZE, TEXT_DIM, DiT


def ddim_sample(model, text, cfg, steps, device, seed=0):
    # DDIM, 50 steps by default, much faster than 1000 DDPM steps
    # and good enough to see what the model has learned
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(text.shape[0], 3, IMAGE_SIZE, IMAGE_SIZE,
                    generator=g, device=device)
    timesteps = torch.linspace(999, 0, steps, dtype=torch.long, device=device)
    alphas_cumprod = torch.cumprod(1.0 - torch.linspace(1e-4, 0.02, 1000), dim=0).to(device)
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(text.shape[0])
            # classifier-free guidance: run with text and without, mix
            text_both = torch.cat([torch.zeros_like(text), text], dim=0)
            x_both = torch.cat([x, x], dim=0)
            t_both = torch.cat([t_batch, t_batch], dim=0)
            pred = model(x_both, t_both, text_both)
            pred_uncond, pred_cond = pred.chunk(2, dim=0)
            eps = pred_uncond + cfg * (pred_cond - pred_uncond)
            a_t = alphas_cumprod[t]
            a_prev = alphas_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0, device=device)
            x0 = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
            x = torch.sqrt(a_prev) * x0 + torch.sqrt(1 - a_prev) * eps
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default="a photo of a cat")
    p.add_argument("--out", default="sample.png")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import CLIPModel, CLIPTokenizer

    clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32").to(device)
    clip_model.eval()

    cfg_path = os.path.join(args.model, "config.json")
    hidden, layers, heads, mlp = 1024, 13, 16, 4096
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            saved = json.load(f)
        hidden = saved.get("hidden", hidden)
        layers = saved.get("layers", layers)
        heads = saved.get("heads", heads)
        mlp = saved.get("mlp", mlp)

    model = DiT(hidden=hidden, layers=layers, heads=heads, mlp=mlp)
    ckpt = torch.load(os.path.join(args.model, "model.pt"),
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    print(f"loaded {args.model}, step {ckpt.get('step', '?')}", flush=True)

    with torch.no_grad():
        ids = clip_tok([args.prompt] * args.n, padding=True,
                       return_tensors="pt").to(device)
        # do not use get_text_features, its return type changed in
        # newer transformers, go through the text model directly
        out = clip_model.text_model(**ids)
        pooled = out.pooler_output
        if pooled is None:
            pooled = out.last_hidden_state[:, 0]
        text = clip_model.text_projection(pooled)
        text = F.normalize(text, dim=-1).to(torch.float32)

    x = ddim_sample(model, text, args.cfg, args.steps, device, seed=args.seed)
    # back to pixels, 0 to 255, tile into a grid
    x = ((x.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8).cpu()
    n = args.n
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    grid = Image.new("RGB", (cols * IMAGE_SIZE, rows * IMAGE_SIZE), "white")
    for i in range(n):
        img = Image.fromarray(x[i].permute(1, 2, 0).numpy())
        grid.paste(img, ((i % cols) * IMAGE_SIZE, (i // cols) * IMAGE_SIZE))
    grid.save(args.out)
    print(f"saved {args.out} ({cols}x{rows} grid, {args.n} images)", flush=True)


if __name__ == "__main__":
    main()
