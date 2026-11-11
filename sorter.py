#!/usr/bin/env python3
"""meme-sorter: 把一堆表情包按内容自动分到几个文件夹里。

用法:
    python sorter.py ./表情包 -o ./分好的
    python sorter.py ./表情包 -o ./分好的 --dry-run
    python sorter.py ./表情包 --classes 开心 卖萌 无语 生气

思路很简单：用 CLIP 做零样本分类，你给几个类名就能分，
不用标注数据、不用训练。拿不准的图统一丢进 unsure。
"""

import argparse
import shutil
import sys
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

DEFAULT_CLASSES = ["开心", "卖萌", "无语", "生气"]

# 每个类给点描述，比只用一个词准得多
PROMPTS = {
    "开心": ["a happy reaction image", "a funny smiling meme", "a laughing sticker"],
    "卖萌": ["a cute kawaii sticker", "an adorable anime character being cute"],
    "无语": ["a speechless facepalm reaction image", "an awkward sighing meme"],
    "生气": ["an angry shouting reaction image", "a furious rage meme"],
}


def collect_images(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def load_clip():
    try:
        import open_clip
        import torch
    except ImportError:
        sys.exit(
            "缺依赖，先装一下:\n"
            "  pip install -r requirements.txt\n"
            "(第一次跑还要下载模型，会慢一点)"
        )
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer, torch


def build_text_features(classes, model, tokenizer, torch):
    import open_clip

    texts = []
    owner = []
    for c in classes:
        for t in PROMPTS.get(c, [f"a meme about {c}"]):
            texts.append(t)
            owner.append(c)
    tok = tokenizer(texts)
    with torch.no_grad():
        feats = model.encode_text(tok)
        feats /= feats.norm(dim=-1, keepdim=True)
    return feats, owner


def classify(path, model, preprocess, text_feats, owner, classes, logit_scale, torch, threshold):
    from PIL import Image

    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return "unsure", 0.0

    with torch.no_grad():
        x = preprocess(img).unsqueeze(0)
        f = model.encode_image(x)
        f /= f.norm(dim=-1, keepdim=True)
        sims = (f @ text_feats.T)[0]

    # 同一类下的多个描述取最高分，再在类别之间做 softmax
    best_per_class = {c: -1.0 for c in classes}
    for score, cls in zip(sims.tolist(), owner):
        if score > best_per_class.get(cls, -1.0):
            best_per_class[cls] = score

    logits = torch.tensor([best_per_class[c] for c in classes]) * logit_scale
    probs = logits.softmax(dim=-1)
    idx = int(probs.argmax())
    score = float(probs[idx])

    if score < threshold:
        return "unsure", score
    return classes[idx], score


def main():
    ap = argparse.ArgumentParser(description="把表情包按内容自动分类")
    ap.add_argument("src", help="图片所在文件夹")
    ap.add_argument("-o", "--out", default=None, help="输出文件夹，默认在原文件夹旁边")
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, help="分类名")
    ap.add_argument("--threshold", type=float, default=0.4,
                    help="最低置信度（0-1），低于这个就丢 unsure")
    ap.add_argument("--dry-run", action="store_true", help="只打印结果，不移动文件")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    if not src.is_dir():
        sys.exit(f"找不到这个文件夹: {src}")

    out = Path(args.out).expanduser().resolve() if args.out else src.parent / (src.name + "_sorted")

    images = list(collect_images(src))
    if not images:
        sys.exit("里面一张图都没有，是不是路径写错了")

    print(f"扫到 {len(images)} 张图，分成 {len(args.classes)} 类 + unsure")

    model, preprocess, tokenizer, torch = load_clip()
    text_feats, owner = build_text_features(args.classes, model, tokenizer, torch)
    logit_scale = float(model.logit_scale.exp()) if hasattr(model, "logit_scale") else 100.0

    stats = {}
    for i, img in enumerate(images, 1):
        cls, score = classify(img, model, preprocess, text_feats, owner,
                              args.classes, logit_scale, torch, args.threshold)
        stats[cls] = stats.get(cls, 0) + 1
        print(f"[{i}/{len(images)}] {img.name} -> {cls} ({score:.2f})")

        if not args.dry_run:
            target = out / cls
            target.mkdir(parents=True, exist_ok=True)
            dest = target / img.name
            n = 1
            while dest.exists():
                dest = target / f"{img.stem}_{n}{img.suffix}"
                n += 1
            shutil.copy2(img, dest)

    print("\n完成:")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
