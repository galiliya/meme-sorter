# meme-sorter

把一堆乱七八糟的表情包，按内容自动分到几个文件夹里。

用 CLIP 做零样本分类，你给类名，它就能分。不用标注数据，不用训练，装完就能跑。

## 为什么

表情包存着存着就几百张，想找「那个卖萌的猫」得翻半天。
手动分类分到第十张就烦了。

## 装

```bash
pip install -r requirements.txt
```

第一次运行会下载 CLIP 模型（约 350MB），之后就不用再下了。

## 用

```bash
# 最省事，默认分 开心 / 卖萌 / 无语 / 生气
python sorter.py ~/Downloads/表情包

# 自己定分类
python sorter.py ./stickers -o ./sorted --classes 猫 狗 无语 生气

# 先看看会怎么分，不动文件
python sorter.py ./stickers --dry-run
```

默认不会动原图，是复制到输出文件夹，分完不满意原文件还在。

## 参数

| 参数 | 说明 |
| --- | --- |
| `src` | 图片文件夹，会递归扫 |
| `-o, --out` | 输出位置，默认 `原文件夹_sorted` |
| `--classes` | 分类名，随便写，中文也行 |
| `--threshold` | 低于这个置信度就丢进 `unsure`，默认 0.4 |
| `--dry-run` | 只打印，不复制 |

## 分不准怎么办

改 `sorter.py` 里的 `PROMPTS`，给每个类多写几条英文描述。
比如「卖萌」写 `a cute kawaii sticker, an adorable anime character being cute`。

CLIP 是拿英文训练的，类名是中文的时候，真正起作用的是这些描述。

拿不准的图会进 `unsure` 文件夹，不会硬猜。

## 支持的格式

jpg / jpeg / png / gif / webp / bmp

动图只按第一帧判断。

## 许可

MIT
