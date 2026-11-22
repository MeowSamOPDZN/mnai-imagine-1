"""Train MNAI Imagine 1 from scratch. Random weights, no base model.

Data is CIFAR-10 first, 50k little 32x32 pictures, resized to 64x64.
Each class gets a plain prompt, "a photo of a cat" and so on, read
by a frozen CLIP text encoder. The DiT learns to undo noise on
those pictures, guided by that text embedding.

Checkpoints save every so often, so a Colab session ending just
means resume with --resume and keep going.

Memory on a free Colab T4 (16GB): weights bf16 0.5GB, grads 0.5GB,
8-bit Adam about 0.25GB, activations with checkpointing a few GB.
That is why 8-bit Adam is the default here, same as Nano 2.

Usage, Colab T4:
  python train.py --out out-im1 --max-images 1000000
  python train.py --out out-im1 --resume out-im1/ckpt-500

Kaggle, stop cleanly inside the 12h cap:
  python train.py --out out-im1 --max-images 1000000 --max-hours 10.5

Smoke test on CPU, no data download. Uses a tiny twin of the same
block layout for the loop test, full 250M forward is in model.py:
  python train.py --smoke --max-steps 3 --out /tmp/smoke-im1
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from model import IMAGE_SIZE, TEXT_DIM, DiT, build_model, count_params

# CIFAR-10 classes, in order, with a plain prompt each
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
PROMPTS = [f"a photo of a {c}" for c in CLASSES]

TIMESTEPS = 1000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="out-im1")
    p.add_argument("--resume", default=None)
    p.add_argument("--max-images", type=int, default=1_000_000)
    p.add_argument("--max-steps", type=int, default=0, help="optimizer steps, 0 = use max-images")
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--optim", default="bnb8bit", choices=["bnb8bit", "adamw"])
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--max-hours", type=float, default=0,
                   help="stop after this many hours, saving a checkpoint first (Kaggle 12h cap)")
    p.add_argument("--guidance-drop", type=float, default=0.1)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def get_betas():
    # linear DDPM schedule, 1000 steps, the plain standard one
    return torch.linspace(1e-4, 0.02, TIMESTEPS)


def q_sample(x0, t, noise, alphas_cumprod):
    # add noise to a clean image, at timestep t
    a = alphas_cumprod[t][:, None, None, None]
    return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise


def encode_prompts(clip_model, clip_tok, device):
    # one CLIP embedding per class, computed once and reused
    # do not use get_text_features, its return type changed in
    # newer transformers, go through the text model directly
    with torch.no_grad():
        ids = clip_tok(PROMPTS, padding=True, return_tensors="pt").to(device)
        out = clip_model.text_model(**ids)
        pooled = out.pooler_output
        if pooled is None:
            pooled = out.last_hidden_state[:, 0]
        text = clip_model.text_projection(pooled)
    return F.normalize(text, dim=-1)


def image_stream(smoke=False):
    # endless stream of (image, class) pairs, so training never waits
    if smoke:
        rng = random.Random(0)
        while True:
            img = torch.randn(3, IMAGE_SIZE, IMAGE_SIZE)
            yield img, rng.randrange(10)
        return
    from torchvision import datasets, transforms

    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tf)
    print(f"cifar10: {len(ds):,} images, 10 classes", flush=True)
    while True:
        idx = list(range(len(ds)))
        random.shuffle(idx)
        for i in idx:
            img, label = ds[i]
            yield img, label


def save_ckpt(model, out_dir, step, images_seen):
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"model": model.state_dict(), "step": step,
                "images_seen": images_seen}, os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"hidden": 1024, "layers": 13, "heads": 16, "mlp": 4096,
                   "image_size": IMAGE_SIZE, "patch": 4, "text_dim": TEXT_DIM,
                   "step": step, "images_seen": images_seen}, f, indent=2)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    betas = get_betas()
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

    train_dtype = torch.float32 if device == "cpu" else torch.bfloat16

    if args.smoke:
        print("smoke mode: random images, no dataset, no CLIP download", flush=True)
        model = DiT(hidden=256, layers=2, heads=4, mlp=512)
        print("smoke model: tiny twin, same block layout, 2 layers", flush=True)
        text_embs = torch.randn(10, TEXT_DIM)
        step = 0
        images_seen = 0
    else:
        from transformers import CLIPModel, CLIPTokenizer

        clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32").to(device)
        clip_model.eval()
        for p_ in clip_model.parameters():
            p_.requires_grad = False
        text_embs = encode_prompts(clip_model, clip_tok, device)
        print(f"text embeddings: {tuple(text_embs.shape)}", flush=True)
        if args.resume:
            print(f"resuming weights from {args.resume}", flush=True)
            model = build_model(dtype=train_dtype)
            ckpt = torch.load(os.path.join(args.resume, "model.pt"),
                              map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"])
            step = ckpt.get("step", 0)
            images_seen = ckpt.get("images_seen", 0)
            print(f"resumed at step {step}, images {images_seen:,}", flush=True)
        else:
            model = build_model(dtype=train_dtype)
            step = 0
            images_seen = 0

    total, _ = count_params(model)
    print(f"model: {total:,} params ({total / 1e6:.2f}M)", flush=True)
    model.to(device)
    model.train()

    if args.smoke:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        print("optimizer: AdamW (smoke, tiny model)", flush=True)
    elif args.optim == "bnb8bit":
        try:
            import bitsandbytes as bnb

            opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay, betas=(0.9, 0.95))
            print("optimizer: AdamW 8-bit", flush=True)
        except Exception as e:
            print(f"8-bit Adam not available ({e}), falling back to AdamW", flush=True)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay, betas=(0.9, 0.95))
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay, betas=(0.9, 0.95))
        print("optimizer: AdamW", flush=True)

    images_per_step = args.micro_batch * args.grad_accum
    if args.max_steps:
        total_steps = args.max_steps
    else:
        total_steps = max(1, args.max_images // images_per_step)
    from transformers import get_cosine_schedule_with_warmup

    sched = get_cosine_schedule_with_warmup(opt, args.warmup, total_steps)
    print(f"plan: {total_steps} optimizer steps, {images_per_step} images/step", flush=True)
    print(f"target images: {args.max_images:,}", flush=True)

    stream = image_stream(smoke=args.smoke)
    use_bf16 = device == "cuda"
    t0 = time.time()
    micro = 0
    loss_sum = 0.0
    loss_n = 0

    while step < total_steps:
        imgs, labels = [], []
        for _ in range(args.micro_batch):
            img, label = next(stream)
            imgs.append(img)
            labels.append(label)
        x0 = torch.stack(imgs).to(device)
        labels_t = torch.tensor(labels, device=device)
        text = text_embs[labels_t].to(device)
        # classifier-free guidance training: drop the text sometimes
        drop = torch.rand(text.shape[0], device=device) < args.guidance_drop
        text = torch.where(drop[:, None], torch.zeros_like(text), text)
        t = torch.randint(0, TIMESTEPS, (x0.shape[0],), device=device)
        noise = torch.randn_like(x0)
        xt = q_sample(x0, t, noise, alphas_cumprod)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
            pred = model(xt, t, text, use_ckpt=(device == "cuda"))
            loss = F.mse_loss(pred, noise) / args.grad_accum
        loss.backward()
        loss_sum += loss.item() * args.grad_accum
        loss_n += 1
        micro += 1
        images_seen += x0.shape[0]

        if micro % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                el = time.time() - t0
                img_s = images_seen / max(el, 1)
                print(f"step {step}/{total_steps} loss {loss_sum / max(loss_n, 1):.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} images {images_seen:,} "
                      f"{img_s:,.1f} img/s elapsed {el / 60:.1f}m", flush=True)
                loss_sum, loss_n = 0.0, 0
            if step % args.ckpt_every == 0 or step == total_steps:
                ckpt = os.path.join(args.out, f"ckpt-{step}")
                print(f"saving {ckpt}", flush=True)
                save_ckpt(model, ckpt, step, images_seen)
                with open(os.path.join(args.out, "train_log.jsonl"), "a") as f:
                    f.write(json.dumps({"step": step, "images_seen": images_seen,
                                        "time_s": time.time() - t0}) + "\n")
            if args.max_hours and time.time() - t0 > args.max_hours * 3600:
                ckpt = os.path.join(args.out, f"ckpt-{step}")
                if not os.path.exists(os.path.join(ckpt, "model.pt")):
                    print(f"time limit reached, saving {ckpt}", flush=True)
                    save_ckpt(model, ckpt, step, images_seen)
                print(f"stopping at step {step}, time limit {args.max_hours}h reached", flush=True)
                break
            if args.smoke and step >= 3:
                print("smoke done, 3 steps ran fine", flush=True)
                break

    print(f"done. steps {step}, images {images_seen:,}, time {(time.time() - t0) / 60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
