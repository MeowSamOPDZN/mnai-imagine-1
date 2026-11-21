# MNAI Imagine 1 — 250M, from scratch

A real image model, built from random weights. Not an API wrapper, not a fine-tune of someone else's image model.

Made by **MeowSamOPDZN Studio**. Owner: **MeowSamOP DZN**. Full form: **Meow x Nythera AI**.

## The model

| thing | value |
|---|---|
| params | **249,920,560 — 250M** |
| shape | DiT, AdaLN-Zero, pixel space |
| hidden | 1024 |
| layers | 13 |
| attention heads | 16 |
| feed-forward | 4096, GELU |
| image | 64x64, 4x4 patches, 256 tokens |
| timesteps | 1000, DDPM, linear beta schedule |
| text | frozen CLIP ViT-B/32, 512-dim pooled, AdaLN conditioning |
| guidance | classifier-free, text dropped 10% in training |

Why this shape: 13 layers at 1024 hidden lands right on 250M, AdaLN carries the timestep and the text in, and pixel space means no VAE to train first. It is the same transformer idea as Nano 2, adapted for images — Nano 2 predicts the next token, this one predicts the noise to remove.

Files: `model.py` is the shape and the param count, `train.py` is the training loop, `generate.py` makes pictures with DDIM sampling, `prompts.txt` is the ten prompts it knows first.

## Honest numbers first

This is a first prototype, so set expectations right.

- It is trained first on CIFAR-10: 50,000 little pictures, 10 classes. It will make a cat, a dog, a car, a bird — simple things, at 64x64, a bit soft and blurry at first.
- It will **not** make photorealistic open-ended images like Qwen-Image-2.1. That one is a 7B DiT plus an 8B text encoder, 16.2B params total, about 33GB of weights, trained by Alibaba on billions of images. This is 250M, trained on 50k images.
- A first useful run is **1M images** (20 CIFAR epochs): roughly 6–10 hours of T4 time, split across sessions, at an estimated 64 images per optimizer step. First recognisable samples should show up after about 100k images.
- Weights bf16 are 0.5GB, 8-bit Adam is ~0.25GB, grads 0.5GB. That is why 8-bit Adam is the default — same as Nano 2.
- Checkpoints save as `model.pt` + `config.json`, so a session ending is not lost work. Resume and keep going.

Your 4GB RAM PC cannot train this. It *can* run a finished checkpoint to make 64x64 pictures, the weights are only 0.5GB.

## Run it in Colab (free T4)

Open `mnai_imagine1_colab.ipynb` in Colab, GPU runtime, T4. It does, in order:

1. Clone this repo, install deps.
2. Check the model: `python model.py`
3. Train: `python train.py --out out-im1 --max-images 1000000`
4. When a session ends, resume: `python train.py --out out-im1 --resume out-im1/ckpt-500`
5. Make pictures: `python generate.py --model out-im1/ckpt-500 --prompt "a photo of a cat" --n 4 --out sample.png`
6. Download `sample.png` from Colab.

Save checkpoints to Google Drive in the notebook if you do not want to re-download them every session. The notebook mounts Drive for that.

## Run it locally (smoke test only)

```bash
python model.py
python train.py --smoke --max-steps 3 --out /tmp/smoke-im1
```

`model.py` proves the full 250M shape and forward pass. The smoke test proves the training loop and checkpoint save on a tiny twin of the same layout, because a full 250M backward pass does not fit in CPU RAM here. Real training needs the Colab GPU.

## Data

CIFAR-10, downloaded by torchvision on first run, resized to 64x64. Plus the frozen CLIP text encoder, `openai/clip-vit-base-patch32`, which only reads prompts — it is not trained here.

## What comes next

Imagine 1.2 is the same block, bigger: about 250M→~500M, 128x128, trained longer on more images. Imagine 5 is the far one: 1–2B, 1024x1024, H100 cluster. This repo is step one.

## License

MIT. The model weights you train are yours.
