"""Kaggle GPU job: extract frozen features once, so everything else runs on CPU.

For every sampled image and every *view* (perturbation), store:
  - CLIP ViT-L/14 image embedding (768-d)        [UnivFD-style representation]
  - DINOv2 ViT-B/14 CLS embedding (768-d)
  - 64-d low-level spectrum feature (radial FFT profile of the noise residual)

Views:
  raw      : file as stored (GenImage: real=JPEG, fake=PNG -> format shortcut!)
  jpeg95   : every image re-encoded JPEG q95 (removes the format shortcut)
  jpeg75, jpeg50, resize50, blur : realistic degradations (test split only)

Output: /kaggle/working/feats/{meta.csv, <model>_<view>.npy, spec_<view>.npy}
"""
from __future__ import annotations

import io
import os
import random
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter

N_TRAIN_PER_CLASS = int(os.getenv("N_TRAIN", 1000))
N_TEST_PER_CLASS = int(os.getenv("N_TEST", 500))
OUT = "/kaggle/working/feats"
os.makedirs(OUT, exist_ok=True)
random.seed(0)


def find_subsets(root="/kaggle/input"):
    subs = {}
    for dp, dns, _ in os.walk(root):
        base = os.path.basename(dp)
        if base.lower().startswith("imagenet_") and {"train", "val"} & set(dns):
            subs[base.split("_")[-1].lower()] = dp
    return subs


def list_imgs(d):
    return sorted(os.path.join(d, f) for f in os.listdir(d)) if os.path.isdir(d) else []


IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")


def build_external_meta(n=int(os.getenv("N_EXT", 500))):
    """Fakes from generators never seen in training (Synthbuster + Nano Banana). All label=1."""
    rows = []
    for dp, dns, fns in os.walk("/kaggle/input"):
        imgs = sorted(f for f in fns if f.lower().endswith(IMG_EXT))
        if len(imgs) < 50:
            continue
        low = dp.lower()
        if "synthbuster" in low:
            gen = "sb_" + os.path.basename(dp).lower()
        elif "nano banana" in low or "nano-banana" in low:
            gen = "nanobanana2_unverified"
        else:
            continue
        random.shuffle(imgs)
        for f in imgs[:n]:
            rows.append({"path": os.path.join(dp, f), "generator": gen, "split": "test", "label": 1,
                         "ext": os.path.splitext(f)[1].lower()})
    meta = pd.DataFrame(rows)
    print(meta.groupby("generator").size(), flush=True)
    return meta


def build_meta():
    if os.getenv("EXTERNAL"):
        return build_external_meta()
    subs = find_subsets()
    print("found subsets:", subs, flush=True)
    rows = []
    for gen, d in sorted(subs.items()):
        for split, n in (("train", N_TRAIN_PER_CLASS), ("val", N_TEST_PER_CLASS)):
            for cls, label in (("ai", 1), ("nature", 0)):
                files = list_imgs(os.path.join(d, split, cls))
                random.shuffle(files)
                for f in files[:n]:
                    rows.append({"path": f, "generator": gen, "split": "train" if split == "train" else "test",
                                 "label": label, "ext": os.path.splitext(f)[1].lower()})
    meta = pd.DataFrame(rows)
    print(meta.groupby(["generator", "split", "label"]).size().unstack(), flush=True)
    print("extensions by label:\n", meta.groupby(["label", "ext"]).size(), flush=True)
    return meta


def jpeg(img, q):
    b = io.BytesIO(); img.save(b, "JPEG", quality=q); b.seek(0)
    return Image.open(b).convert("RGB")


VIEWS = {
    "raw": lambda im: im,
    "jpeg95": lambda im: jpeg(im, 95),
    "jpeg75": lambda im: jpeg(im, 75),
    "jpeg50": lambda im: jpeg(im, 50),
    "resize50": lambda im: jpeg(im.resize((max(1, im.width // 2), max(1, im.height // 2)), Image.BICUBIC), 95),
    "blur": lambda im: jpeg(im.filter(ImageFilter.GaussianBlur(1.0)), 95),
}
TEST_ONLY = {"jpeg75", "jpeg50", "resize50", "blur"}


def center_crop(im, s=224):
    if min(im.size) < s:
        r = s / min(im.size)
        im = im.resize((round(im.width * r), round(im.height * r)), Image.BICUBIC)
    l, t = (im.width - s) // 2, (im.height - s) // 2
    return im.crop((l, t, l + s, t + s))


def spectrum_feat(im, bins=64):
    g = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    # noise residual: image minus 3x3 box blur, emphasises generator upsampling artifacts
    k = np.ones((3, 3), np.float32) / 9
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.pad(g, 1, mode="reflect")
    res = g - (sliding_window_view(pad, (3, 3)) * k).sum((-1, -2))
    f = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(res))))
    h, w = f.shape
    yy, xx = np.indices(f.shape)
    r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    r = (r / r.max() * (bins - 1)).astype(int)
    prof = np.bincount(r.ravel(), f.ravel(), bins) / np.maximum(np.bincount(r.ravel(), minlength=bins), 1)
    return prof.astype(np.float32)


MEAN = {"clip": (0.4815, 0.4578, 0.4082), "dino": (0.485, 0.456, 0.406)}
STD = {"clip": (0.2686, 0.2613, 0.2758), "dino": (0.229, 0.224, 0.225)}


def to_tensor(crops, kind, dev):
    x = torch.from_numpy(np.stack([np.asarray(c, np.float32) / 255.0 for c in crops])).permute(0, 3, 1, 2)
    m = torch.tensor(MEAN[kind]).view(1, 3, 1, 1); s = torch.tensor(STD[kind]).view(1, 3, 1, 1)
    return ((x - m) / s).to(dev, torch.float16)


def main():
    from transformers import AutoModel, CLIPVisionModelWithProjection
    meta = build_meta()
    if meta.empty:
        print("NO DATA FOUND under /kaggle/input:", os.listdir("/kaggle/input")); sys.exit(1)
    meta.to_csv(f"{OUT}/meta.csv", index=False)
    d0 = "cuda:0"; d1 = "cuda:1" if torch.cuda.device_count() > 1 else d0
    clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to(d0).eval()
    dino = AutoModel.from_pretrained("facebook/dinov2-base", torch_dtype=torch.float16).to(d1).eval()

    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(8)

    def load(p, view):
        im = Image.open(p).convert("RGB")
        im = center_crop(VIEWS[view](im))  # perturb at native size, then crop
        return im, spectrum_feat(im)

    only_views = [v for v in os.getenv("ONLY_VIEWS", "").split(",") if v]
    only_split = os.getenv("ONLY_SPLIT")
    for view in (only_views or VIEWS):
        if only_split:
            idx = meta.index[meta.split == only_split]
        else:
            idx = meta.index if view not in TEST_ONLY else meta.index[meta.split == "test"]
        paths = meta.loc[idx, "path"].tolist()
        C = np.zeros((len(meta), 768), np.float16); D = np.zeros((len(meta), 768), np.float16)
        S = np.zeros((len(meta), 64), np.float32); have = np.zeros(len(meta), bool)
        t = time.time(); bs = 128
        for i in range(0, len(paths), bs):
            res = list(pool.map(lambda p: load(p, view), paths[i:i + bs]))
            crops = [r[0] for r in res]
            rows = idx[i:i + bs]
            with torch.no_grad():
                C[rows] = clip(pixel_values=to_tensor(crops, "clip", d0)).image_embeds.float().cpu().numpy()
                D[rows] = dino(pixel_values=to_tensor(crops, "dino", d1)).pooler_output.float().cpu().numpy()
            S[rows] = np.stack([r[1] for r in res]); have[rows] = True
            if i % (bs * 20) == 0:
                print(f"{view} {i}/{len(paths)} {time.time()-t:.0f}s", flush=True)
        sfx = f"_{only_split}" if only_split else ""
        np.save(f"{OUT}/clip_{view}{sfx}.npy", C); np.save(f"{OUT}/dino_{view}{sfx}.npy", D)
        np.save(f"{OUT}/spec_{view}{sfx}.npy", S); np.save(f"{OUT}/have_{view}{sfx}.npy", have)
        print(f"done {view}: {have.sum()} imgs in {time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
