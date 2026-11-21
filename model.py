"""MNAI Imagine 1, the 250M image model. Own shape, trained from scratch.

A diffusion transformer, AdaLN-Zero style, in pixel space at 64x64.
It learns to undo noise, step by step, guided by a text embedding.

The shape is ours: 1024 hidden, 13 layers, 16 heads, a 4096-wide
feed-forward, 4x4 patches, so a 64x64 image becomes 256 tokens.
It adds up to 249,920,560 parameters, so 0.25B, call it 250M.

Why this shape: 13 layers at 1024 hidden lands right on 250M,
the AdaLN modulation carries the timestep and the text in, and
pixel space means no VAE to train first. The text side is a frozen
CLIP text encoder, it only reads the prompt, it never draws.

That is the honest limit too. At 64x64, trained first on CIFAR-10,
it learns simple things well, a cat, a car, a bird, not open-ended
photoreal images. That is what a first 250M can do. Bigger versions
grow from this same block.
"""
import math
import os

import torch
import torch.nn as nn

# the shape, all in one place so train.py and generate.py agree
IMAGE_SIZE = 64
PATCH_SIZE = 4
IN_CHANNELS = 3
HIDDEN_SIZE = 1024
NUM_LAYERS = 13
NUM_HEADS = 16
MLP_SIZE = 4096
TEXT_DIM = 512
TIME_DIM = 256
NUM_TOKENS = (IMAGE_SIZE // PATCH_SIZE) ** 2


def timestep_embedding(t, dim=TIME_DIM, max_period=10000):
    # sinusoidal timesteps, same idea as the original DiT paper
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class DiTBlock(nn.Module):
    # one transformer block, AdaLN-Zero: the timestep and text
    # decide the scale, shift and gate for both halves of the block
    def __init__(self, hidden=HIDDEN_SIZE, heads=NUM_HEADS, mlp=MLP_SIZE):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp),
            nn.GELU(),
            nn.Linear(mlp, hidden),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))

    def forward(self, x, c):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(c).chunk(6, dim=1)
        # attention half, modulated then gated
        h = self.norm1(x) * (1 + scale1[:, None, :]) + shift1[:, None, :]
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate1[:, None, :] * h
        # feed-forward half, same trick
        h = self.norm2(x) * (1 + scale2[:, None, :]) + shift2[:, None, :]
        h = self.mlp(h)
        return x + gate2[:, None, :] * h


class DiT(nn.Module):
    def __init__(self, hidden=HIDDEN_SIZE, layers=NUM_LAYERS,
                 heads=NUM_HEADS, mlp=MLP_SIZE, image_size=IMAGE_SIZE,
                 patch=PATCH_SIZE, in_ch=IN_CHANNELS, text_dim=TEXT_DIM):
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        self.patch = patch
        self.in_ch = in_ch
        self.image_size = image_size
        n_tokens = (image_size // patch) ** 2
        self.n_tokens = n_tokens
        # patches in, a linear layer over each flattened 4x4 patch
        self.patch_embed = nn.Linear(in_ch * patch * patch, hidden)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, hidden))
        # time goes through a small MLP, text is projected to match
        self.time_mlp = nn.Sequential(
            nn.Linear(TIME_DIM, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.text_proj = nn.Linear(text_dim, hidden)
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden, heads, mlp) for _ in range(layers)]
        )
        # final norm and head, back out to patches
        self.norm_final = nn.LayerNorm(hidden)
        self.ada_final = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        self.head = nn.Linear(hidden, in_ch * patch * patch)
        self._init_weights()

    def _init_weights(self):
        # keep the start calm, AdaLN gates start at zero so each
        # block begins as an identity, the usual DiT trick
        nn.init.normal_(self.pos_embed, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.ada[1].weight, 0)
            nn.init.constant_(block.ada[1].bias, 0)
        nn.init.constant_(self.ada_final[1].weight, 0)
        nn.init.constant_(self.ada_final[1].bias, 0)
        nn.init.constant_(self.head.weight, 0)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x, t, text_emb, use_ckpt=False):
        # x: (B, 3, 64, 64) in [-1, 1], t: (B,) timesteps,
        # text_emb: (B, 512) frozen CLIP pooled embedding
        b = x.shape[0]
        p = self.patch
        # cut into patches, flatten each one, project it
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(b, self.n_tokens, -1)
        x = self.patch_embed(x) + self.pos_embed
        # one conditioning vector, time plus text
        c = self.time_mlp(timestep_embedding(t).to(x.dtype)) + self.text_proj(text_emb.to(x.dtype))
        for block in self.blocks:
            if use_ckpt:
                x = torch.utils.checkpoint.checkpoint(block, x, c, use_reentrant=False)
            else:
                x = block(x, c)
        shift, scale = self.ada_final(c).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale[:, None, :]) + shift[:, None, :]
        x = self.head(x)
        # patches back out to a full image, same shape as the input
        x = x.reshape(b, self.image_size // p, self.image_size // p,
                      self.in_ch, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(b, self.in_ch, self.image_size, self.image_size)
        return x


def build_model(dtype=torch.bfloat16):
    model = DiT()
    if dtype is not None:
        model = model.to(dtype)
    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def describe(model):
    total, _ = count_params(model)
    lines = [
        f"image {IMAGE_SIZE}x{IMAGE_SIZE}, patch {PATCH_SIZE}, tokens {NUM_TOKENS}",
        f"hidden {HIDDEN_SIZE}, layers {NUM_LAYERS}, heads {NUM_HEADS}, mlp {MLP_SIZE}",
        f"text dim {TEXT_DIM}, time dim {TIME_DIM}",
        f"params {total:,} ({total / 1e6:.2f}M)",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    model = build_model(dtype=torch.float32)
    print(describe(model))
    print()
    # tiny forward pass, just to prove the shape runs
    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    t = torch.tensor([100])
    text = torch.randn(1, TEXT_DIM)
    out = model(x, t, text)
    print("output shape:", tuple(out.shape))
    print("expected:", (1, 3, IMAGE_SIZE, IMAGE_SIZE))
