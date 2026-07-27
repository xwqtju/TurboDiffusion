# B200 上安装 VBench 并评测 Original / SLA / SLA + Q 2:4

本文面向没有 AI Agent 辅助的 B200 服务器，命令可以逐段复制执行。评测使用官方
[VBench](https://github.com/Vchitect/VBench) 代码和预训练评分模型。生成模型权重、生成视频和
VBench 评分权重都不会提交到 Git。

## 1. 本次评测范围

自定义视频输入模式正式支持以下六项：

- `subject_consistency`
- `background_consistency`
- `motion_smoothness`
- `dynamic_degree`
- `aesthetic_quality`
- `imaging_quality`

六项的算术平均只能作为本实验内部的汇总值，**不是**完整 16 维 VBench 官方 Total。

## 2. 系统依赖

以下环境已验证可用：Python 3.12、PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128。
B200 属于 Blackwell GPU，不要照搬 VBench README 中较旧的 CUDA 11.8 安装命令。

```bash
sudo apt update
sudo apt install -y git wget unzip ffmpeg jq python3-venv

nvidia-smi
python3 --version
```

确认驱动能够识别 B200。建议把 VBench 放在 TurboDiffusion 同级目录：

```text
workspace/
├── TurboDiffusion/
└── VBench/
```

## 3. 安装正式 VBench 包

如果 TurboDiffusion 环境中已经有可用的 PyTorch 2.8 + CUDA 12.8，可复用其
site-packages，避免再次下载数 GB 的 PyTorch：

```bash
cd /path/to/workspace
git clone https://github.com/Vchitect/VBench.git
cd VBench

/path/to/turbodiffusion-env/bin/python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

如果没有现成的 PyTorch 环境，建立独立环境：

```bash
cd /path/to/workspace
git clone https://github.com/Vchitect/VBench.git
cd VBench

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

本实验的六项自定义指标不需要 Detectron2。不要仅为了这六项安装 Detectron2，旧版本
Detectron2 对 CUDA 12.8/Blackwell 的兼容性较差。

验证安装：

```bash
python - <<'PY'
import torch
import vbench
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
assert torch.cuda.is_available()
PY
```

## 4. 下载六项指标需要的评分模型

统一把评分权重放在 `VBench/checkpoints`：

```bash
cd /path/to/workspace/VBench
source .venv/bin/activate

export VBENCH_CACHE_DIR="$PWD/checkpoints"
mkdir -p \
    "$VBENCH_CACHE_DIR/clip_model" \
    "$VBENCH_CACHE_DIR/amt_model" \
    "$VBENCH_CACHE_DIR/raft_model" \
    "$VBENCH_CACHE_DIR/dino_model" \
    "$VBENCH_CACHE_DIR/aesthetic_model/emb_reader" \
    "$VBENCH_CACHE_DIR/pyiqa_model"
```

### 4.1 CLIP：背景一致性和美学质量

```bash
wget -c -O "$VBENCH_CACHE_DIR/clip_model/ViT-B-32.pt" \
  https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt

wget -c -O "$VBENCH_CACHE_DIR/clip_model/ViT-L-14.pt" \
  https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
```

### 4.2 DINO：主体一致性

本地模式既需要 DINO 权重，也需要 DINO 源码目录：

```bash
git clone https://github.com/facebookresearch/dino \
  "$VBENCH_CACHE_DIR/dino_model/facebookresearch_dino_main"

wget -c -O "$VBENCH_CACHE_DIR/dino_model/dino_vitbase16_pretrain.pth" \
  https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth
```

### 4.3 AMT-S：运动平滑度

```bash
wget -c -O "$VBENCH_CACHE_DIR/amt_model/amt-s.pth" \
  https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth
```

### 4.4 RAFT：动态程度

```bash
wget -c -O "$VBENCH_CACHE_DIR/raft_model/models.zip" \
  https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
unzip -o "$VBENCH_CACHE_DIR/raft_model/models.zip" \
  -d "$VBENCH_CACHE_DIR/raft_model"
```

解压后必须存在：

```text
checkpoints/raft_model/models/raft-things.pth
```

### 4.5 LAION aesthetic predictor：美学质量

```bash
wget -c -O "$VBENCH_CACHE_DIR/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth" \
  https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_l_14_linear.pth
```

### 4.6 MUSIQ-SPAQ：成像质量

```bash
wget -c -O "$VBENCH_CACHE_DIR/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth" \
  https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth
```

下载完成后检查，任何一行显示 `MISSING` 都不能开始评分：

```bash
for path in \
  clip_model/ViT-B-32.pt \
  clip_model/ViT-L-14.pt \
  dino_model/facebookresearch_dino_main/hubconf.py \
  dino_model/dino_vitbase16_pretrain.pth \
  amt_model/amt-s.pth \
  raft_model/models/raft-things.pth \
  aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth \
  pyiqa_model/musiq_spaq_ckpt-358bb6af.pth
do
  test -s "$VBENCH_CACHE_DIR/$path" && echo "OK $path" || echo "MISSING $path"
done
```

`VBENCH_CACHE_DIR` 每次打开新 shell 后都要重新设置，或者写入任务启动脚本。

## 5. 准备 TurboDiffusion 评测输入

```bash
cd /path/to/workspace/TurboDiffusion
source /path/to/turbodiffusion-env/bin/activate
```

提示词文件已经包含在仓库根目录的 `vbench_prompt.json`。I2V 还需要一张与每个视频同名的条件图：

```text
input_images/
├── creature.png
├── garden.png
├── mammoths.png
├── monster.png
├── pigeon.png
├── televisions.png
├── waves.png
└── woman.png
```

扩展名可为 `.png`、`.jpg`、`.jpeg` 或 `.webp`，文件 stem 必须和 JSON 中 MP4 的
stem 一致。

确认非量化 14B 权重存在：

```bash
test -s checkpoints/TurboWan2.2-I2V-A14B-high-720P.pth
test -s checkpoints/TurboWan2.2-I2V-A14B-low-720P.pth
test -s checkpoints/Wan2.1_VAE.pth
test -s checkpoints/models_t5_umt5-xxl-enc-bf16.pth
```

## 6. 生成三组配对视频

B200 显存足以让每张卡独立运行一个 14B 任务。下面的脚本会把任务分配到 GPU 0、1，
并保持提示词、输入图、seed、步数、帧数和分辨率一致：

```bash
cd /path/to/workspace/TurboDiffusion
source /path/to/turbodiffusion-env/bin/activate

python scripts/run_vbench_attention_ablation.py \
  --pipeline i2v-a14b \
  --prompts vbench_prompt.json \
  --input-image-dir /path/to/input_images \
  --high-noise-checkpoint checkpoints/TurboWan2.2-I2V-A14B-high-720P.pth \
  --low-noise-checkpoint checkpoints/TurboWan2.2-I2V-A14B-low-720P.pth \
  --output-dir output/vbench_attention_ablation_14b_720p81 \
  --gpus 0,1 \
  --methods original sla sla_q_2to4 \
  --seed 0 \
  --num-steps 4 \
  --num-frames 81 \
  --resolution 720p \
  --sla-topk 0.1
```

脚本默认跳过已有的非空视频，任务中断后执行同一条命令即可续跑。只有明确希望覆盖结果时才加
`--overwrite`。输出结构为：

```text
output/vbench_attention_ablation_14b_720p81/
├── original/*.mp4
├── sla/*.mp4
├── sla_q_2to4/*.mp4
├── config.json
└── manifest.jsonl
```

> 如果使用两张 32 GB GPU 而不是 B200，请使用 README 中的
> `wan2.2_i2v_dist_infer.py` 双卡 FSDP2 + context-parallel 命令，不能让上面的单卡任务调度器
> 直接加载非量化 14B。

## 7. 运行正式 VBench 六项评分

以下命令逐组评分，最稳妥，也最容易定位失败组：

```bash
cd /path/to/workspace/VBench
source .venv/bin/activate

export VBENCH_CACHE_DIR="$PWD/checkpoints"
export TURBO_ROOT=/path/to/workspace/TurboDiffusion
export VIDEO_ROOT="$TURBO_ROOT/output/vbench_attention_ablation_14b_720p81"
export DIMS="subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality"

for METHOD in original sla sla_q_2to4; do
  CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29710 python evaluate.py \
    --videos_path "$VIDEO_ROOT/$METHOD" \
    --dimension $DIMS \
    --mode custom_input \
    --prompt_file "$TURBO_ROOT/vbench_prompt.json" \
    --load_ckpt_from_local True \
    --output_path "$VIDEO_ROOT/vbench_scores/$METHOD"
done
```

也可以让两组同时占用两张 B200，但必须给不同进程设置不同 `MASTER_PORT`。

结果文件位于：

```text
vbench_scores/<method>/results_<timestamp>_eval_results.json
```

每个维度的 JSON 值是 `[聚合分数, 逐视频明细]`。只查看聚合分数：

```bash
for FILE in "$VIDEO_ROOT"/vbench_scores/*/*_eval_results.json; do
  echo "$FILE"
  jq 'with_entries(.value = .value[0])' "$FILE"
done
```

## 8. 常见问题

### `CUDA capability sm_100 is not compatible`

安装了不支持 B200 的旧 PyTorch。重新安装 CUDA 12.8 构建：

```bash
python -m pip install --force-reinstall torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

### VBench 又开始联网下载模型

通常是 `VBENCH_CACHE_DIR` 没有设置，或目录层级不正确。重新执行第 4 节的文件检查。

### `No module named vbench`

确认当前位于 VBench 环境，并重新安装本地包：

```bash
cd /path/to/workspace/VBench
source .venv/bin/activate
python -m pip install -e . --no-deps
```

### 某个指标 OOM

先确认每个评分进程只看到一张 GPU。不要在同一张卡上同时启动多个 VBench 任务：

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py ...
```

### `dynamic_degree` 分数只有少数几个离散值

该指标对每个视频先判定“动态/静态”，再对视频集合取平均。只有 8 个视频时分数间隔为
`1/8 = 0.125`，应扩大测试集后再判断细小差异。

### 六项均值下降，但一致性提高

逐项查看分数。SLA 可能提高主体、背景一致性和运动平滑度，同时降低 dynamic degree。
不要只用六项简单平均判断质量，也不要把它写成官方 VBench Total。
