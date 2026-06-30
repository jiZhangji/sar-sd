# Stage 1：多数据集 OPT-to-SAR 基础预训练

本工程当前只实现三阶段路线中的第一阶段：使用大规模配准 OPT-SAR 图像对训练通用跨模态基础模型。此阶段没有 layout、bbox、ROI 或检测损失。Stage 2 将在 M4 上做无 layout 域适配，Stage 3 才输入 `SAR_1 + layout` 做局部残差扩充。

## 服务器一键启动

项目默认适配以下服务器路径：

```text
数据：/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/dataset
代码：/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/sar-sd
```

先用每个数据集100对验证预处理和训练入口：

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/sar-sd
chmod +x run_stage1.sh
LIMIT_PER_DATASET=100 bash run_stage1.sh
```

确认无误后处理全部数据并启动正式预训练：

```bash
LIMIT_PER_DATASET=0 bash run_stage1.sh
```

首次训练会通过Diffusers下载并缓存 `stabilityai/sd-vae-ft-mse` 和 Stable Diffusion v1.5 UNet。若服务器不能访问Hugging Face，请提前下载权重并把 `configs/stage1_opt2sar_pretrain.yaml` 中的 `vae_path`、`unet_path` 改为本地绝对路径。

全量预处理会解压/抽取大量文件，需要为SAR-1M和WHU额外预留足够磁盘空间与inode。建议一定先运行 `LIMIT_PER_DATASET=100`。

脚本会自动：

1. 从 `SAR-1M_full/SAR-1M_DATA.zip` 的 `paired.json` 抽取配对数据；
2. 解压并配对 `whu_opt_sar/optical.zip` 与 `sar.zip`，忽略标签ZIP；
3. 使用 `SAR2Opt_full/A` 和 `B`，忽略重复的Pix2Pix拼接目录；
4. 明确跳过 `M4-SAR`，将其留给Stage 2；
5. 在 `dataset/stage1_prepared/` 写入统一图像、manifest和统计文件；
6. 将manifest传给 `train_stage1.py` 启动训练。

自定义路径：

```bash
DATASET_ROOT=/path/to/dataset \
PROJECT_ROOT=/path/to/sar-sd \
PREPARED_ROOT=/path/to/prepared \
OUTPUT_DIR=/path/to/runs/stage1 \
bash run_stage1.sh
```

断点续训：

```bash
bash run_stage1.sh --resume-from /path/to/checkpoints/last.pt
```

## 1. 当前模型

```text
SAR → 冻结 VAE Encoder → clean latent z0 → DDPM加噪得到zt
OPT → Optical Encoder ─────────────────────────────┐
GSD/传感器/极化/入射角 → Metadata Encoder ────────┤
zt + timestep + OPT tokens + metadata token → UNet → 预测噪声
```

损失严格保持两项：

```text
L = L_diffusion + adaptive_lambda_phy * L_physical
```

- `L_diffusion`：标准噪声预测MSE，始终是主损失。
- `L_physical`：一个辅助损失，内部只含局部均值、局部方差和局部变异系数。
- `adaptive_lambda_phy`：根据两项损失对UNet输出层的梯度比例自适应更新，并带warmup、EMA和硬上限。

详细原理见 `docs/03_多数据集异质性与自适应物理损失.md`。

## 2. 目录

```text
configs/
  stage1_opt2sar_pretrain.yaml       正式配置
  stage1_multidataset_smoke.yaml     CPU/小模型冒烟配置
data/                                本地数据（被.gitignore排除）
docs/                                设计说明
lc_osar/data/paired.py               manifest数据集与平衡采样
lc_osar/models/opt2sar_ldm.py        VAE、OPT/metadata编码器、UNet
lc_osar/losses.py                    GSD感知物理项和自适应控制器
tools/download_datasets.py           HF/ModelScope/官方源下载器
tools/prepare_dataset.py             数据集转统一manifest
tools/merge_manifests.py             合并多个manifest
train_stage1.py                      训练
infer_stage1.py                      DDIM推理
```

数据、权重和运行结果均被 `.gitignore` 排除，可直接将本目录发布到GitHub。

## 3. 安装

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

正式训练需安装与服务器CUDA匹配的PyTorch。

## 4. 数据下载

```powershell
python tools\download_datasets.py list
python tools\download_datasets.py download sar2opt --output D:\opt2sar_raw --limit 2
python tools\download_datasets.py download sen1_2 --output D:\opt2sar_raw --limit 2
```

下载器当前支持：

- SEN1-2：从TUM官方FTP按样本下载；
- SAR2Opt：从Hugging Face按文件下载，可用 `--limit`；
- WHU-OPT-SAR：从Hugging Face社区镜像下载完整ZIP，约14.9GB，必须加 `--full`；
- SAR-1M：从Hugging Face下载76.6GB整包，必须加 `--full`，可能需要 `--token`；
- SOMA-1M test：从作者公开的Google Drive下载，必须加 `--full`；
- 任意ModelScope数据集：使用 `modelscope` 子命令和真实仓库ID。

完整大包示例：

```powershell
python tools\download_datasets.py download whu_opt_sar --output D:\opt2sar_raw --full
python tools\download_datasets.py download sar1m --output D:\opt2sar_raw --full --token $env:HF_TOKEN
python tools\download_datasets.py download soma1m_test --output D:\opt2sar_raw --full
```

ModelScope通用下载：

```powershell
pip install modelscope
python tools\download_datasets.py modelscope namespace/dataset-name --output D:\opt2sar_raw\dataset-name
```

QXS-SAROPT和3MOS目前没有经过核验的HF/ModelScope直接公开镜像，下载器会生成访问提示并指向作者官方仓库，不会使用来源不明的转载。所有数据集的转换命令和元数据说明见 `docs/DATASETS.md`。

## 5. Manifest格式

训练使用JSONL或CSV，而不是只靠同名目录：

```json
{"id":"sen12_001","split":"train","dataset":"SEN1-2","opt_path":"SEN1-2/opt/sen12_001.png","sar_path":"SEN1-2/sar/sen12_001.png","opt_sensor":"Sentinel-2","sar_sensor":"Sentinel-1","opt_gsd":10.0,"sar_gsd":10.0,"polarization":"VV","incidence_angle":null,"sar_unit":"display_uint8"}
```

必填字段：`id`、`split`、`opt_path`、`sar_path`。建议填写数据集、传感器、GSD、极化、入射角、SAR产品和强度单位。未知信息明确写 `unknown`，不要猜测。

正式划分必须按原始场景或地理区域隔离，不能随机拆分相邻切片。

## 6. 多数据集处理

### 分辨率

模型将 `log(opt_gsd)` 和 `log(sar_gsd)` 的Fourier特征作为metadata token。局部统计窗口由米转换为像素：

```text
window_px = odd_clip(round(window_meters / sar_gsd), min_px, max_px)
```

这不会替代预处理。正式数据仍应先配准到共同网格，并按实际地面覆盖范围裁剪，避免把1m和10m数据无脑resize成同一语义尺度。

### 数据平衡

采样概率为 `p(dataset) ∝ N_dataset^temperature`：

- `temperature=1`：按样本量；
- `temperature=0`：各数据集等概率；
- 正式配置默认0.5。

### 辐射值

当前小样本为8位展示图，标记为 `display_uint8`，只用于跑通。正式训练应优先把原始SAR定标为 `sigma0/gamma0 dB`，使用训练集确定的传感器级固定裁剪范围，禁止逐图Min-Max后再声称物理统计可比较。

## 7. 自适应物理权重

正式配置：

```yaml
loss:
  adaptive_physical: true
  physical_target_grad_ratio: 0.05
  physical_lambda_min: 0.0
  physical_lambda_max: 0.10
  physical_lambda_ema: 0.95
  physical_update_interval: 40
  physical_warmup_steps: 5000
  physical_max_timestep: 500
  physical_terms: [mean, variance, cv]
  physical_window_meters: 50.0
```

前5000步不启用物理项；之后只在中低噪声时间步计算。控制器目标是让物理项梯度约为扩散梯度的5%，并确保权重不超过0.1。`lambda_physical`仍作为固定权重模式和控制器初值。

推荐消融：

1. `adaptive_physical: false, lambda_physical: 0`；
2. 固定 `lambda_physical: 0.05`；
3. 自适应，`physical_terms: [mean, variance]`；
4. 自适应，`physical_terms: [mean, variance, cv]`。

## 8. 冒烟训练与推理

已验证命令：

```powershell
python train_stage1.py --config configs\stage1_multidataset_smoke.yaml
python infer_stage1.py --config configs\stage1_multidataset_smoke.yaml --checkpoint runs\stage1_multidataset_smoke\checkpoints\last.pt --manifest data\multidataset_smoke\manifest.jsonl --output runs\stage1_multidataset_smoke\samples --steps 2 --batch-size 2 --precision fp32
```

冒烟配置使用64×64输入和从头构建的小UNet，只验证代码，不代表生成质量。

## 9. 正式训练

准备本地SD Diffusers UNet和VAE，修改：

```yaml
model:
  vae_path: /path/to/vae
  unet_path: /path/to/stable-diffusion
  unet_from_scratch: false
data:
  manifest: /path/to/train_manifest.jsonl
```

启动：

```powershell
python train_stage1.py --config configs\stage1_opt2sar_pretrain.yaml
```

输出包括 `config.json`、`metrics.jsonl`、`checkpoints/last.pt` 和按epoch保存的checkpoint。checkpoint保存模型、优化器、AMP scaler、epoch、global step及自适应控制器状态。

正式配置默认训练10个epoch，每50个batch记录一次TensorBoard标量，每个epoch更新 `last.pt`，并在第2、4、6、8、10个epoch保存编号权重。建议在第5轮做中期评估；第10轮仍有明显改善时，再从 `last.pt` 续训到15至20轮。查看日志：

第2、4、6、8、10轮保存权重后，训练器还会用固定的前4个验证样本执行50步DDIM推理。每个样本的OPT、真实SAR和生成SAR保存到 `runs/stage1_opt2sar_pretrain/samples/epoch_XXXX/`，生成结果也会写入TensorBoard。

```bash
tensorboard --logdir runs/stage1_opt2sar_pretrain/tensorboard --host 0.0.0.0 --port 6006
```

断点续训：

```yaml
train:
  resume_from: runs/stage1_opt2sar_pretrain/checkpoints/last.pt
```

## 10. 正式推理

推理也使用manifest，以便提供目标SAR传感器、GSD和极化条件；`sar_path`可以省略：

```powershell
python infer_stage1.py --config configs\stage1_opt2sar_pretrain.yaml --checkpoint runs\stage1_opt2sar_pretrain\checkpoints\last.pt --manifest data\inference_manifest.jsonl --output outputs\stage1 --steps 50 --batch-size 1 --precision fp16 --seed 42
```

输出为同名单通道PNG。

## 11. 当前边界

当前元数据通过cross-attention token注入，还不是完整的Scale-MAE位置编码或DOFA动态卷积；OPT编码器仍是轻量单尺度CNN。下一步应先做SAR VAE重建检查，再决定是否增加多尺度OPT特征注入。Stage 2和Stage 3尚未混入本工程。

## 12. 审计全量预处理结果

审计脚本会从原始 ZIP/A/B 目录重新计算可用配对，并检查 manifest 是否完整收录、路径是否存在、ID/配对是否重复、划分是否合法，以及 M4 是否被误加入第一阶段：

```bash
python tools/audit_stage1_data.py \
  --dataset-root /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/dataset \
  --prepared-root /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/dataset/stage1_prepared
```

终端和 `stage1_prepared/audit_report.json` 会输出 `PASS` 或 `FAIL`。`FAIL` 时返回非零退出码，可用于训练前自动阻止。SAR2Opt 不需要复制到 `stage1_prepared`；manifest 正确引用原始 A/B 文件即视为已包含。

## 13. 服务器全量训练、输入输出和断点续训

### 13.1 输入和输出

默认服务器原始数据目录：

```text
/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/dataset
```

第一阶段训练输入是 `dataset/stage1_prepared/manifest.jsonl`。manifest 应包含 SAR-1M、WHU-OPT-SAR 和 SAR2Opt，不应包含留给第二阶段的 M4-SAR。

默认输出：

```text
runs/stage1_opt2sar_pretrain/config.json
runs/stage1_opt2sar_pretrain/metrics.jsonl
runs/stage1_opt2sar_pretrain/tensorboard/
runs/stage1_opt2sar_pretrain/checkpoints/last.pt
runs/stage1_opt2sar_pretrain/checkpoints/epoch_0002.pt
runs/stage1_opt2sar_pretrain/samples/epoch_0002/
```

### 13.2 安装和数据审计

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/sar-sd
pip install -r requirements.txt

python tools/audit_stage1_data.py \
  --dataset-root /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/dataset \
  --prepared-root /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/dataset/stage1_prepared
```

只有审计输出 `PASS` 后再启动全量训练。

### 13.3 全量训练

不重新预处理数据，直接使用已审计的 manifest：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train_stage1.py \
  --config configs/stage1_opt2sar_pretrain.yaml \
  --manifest /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/dataset/stage1_prepared/manifest.jsonl \
  --output-dir runs/stage1_opt2sar_pretrain \
  --batch-size 16 \
  --epochs 10
```

`--batch-size` 是每张GPU的 batch size，也可在 `configs/stage1_opt2sar_pretrain.yaml` 的 `data.batch_size` 修改。默认每卡16，两张H100的全局batch为 `16 × 2 × grad_accum_steps(1) = 32`。先用16跑500至1000步；如果包含物理损失的步骤仍有充足显存，再测试每卡24（全局48）。不建议一开始就用32，因为每4步的物理损失需要带梯度的VAE解码，峰值显存高于普通扩散步骤。batch也并非越大越好：大batch通常提高吞吐，但可能降低泛化，还会改变合适的学习率。

训练器使用PyTorch DDP：两卡的加权数据采样会先生成全局序列再按rank分片；只有rank 0写入日志、checkpoint和推理图；自适应物理损失权重会在两卡之间同步。默认使用更适合H100的BF16。

### 13.4 训练时查看

终端进度条会实时显示 `loss`、`diffusion`、`physical`、`lambda_phy` 和 `lr`。另开一个终端启动 TensorBoard：

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/sar-sd
tensorboard --logdir runs/stage1_opt2sar_pretrain/tensorboard --host 0.0.0.0 --port 6006
```

查看每轮汇总日志和最新生成图：

```bash
tail -f runs/stage1_opt2sar_pretrain/metrics.jsonl
find runs/stage1_opt2sar_pretrain/samples -maxdepth 2 -type f | tail -n 20
```

### 13.5 10轮后断点续训

`--epochs` 表示从第1轮开始计算的“总目标轮数”，不是额外轮数。例如已练完10轮，希望续训到20轮：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train_stage1.py \
  --config configs/stage1_opt2sar_pretrain.yaml \
  --manifest /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/dataset/stage1_prepared/manifest.jsonl \
  --output-dir runs/stage1_opt2sar_pretrain \
  --resume-from runs/stage1_opt2sar_pretrain/checkpoints/last.pt \
  --batch-size 16 \
  --epochs 20
```

当前默认学习率是固定的 `1e-5`，没有学习率调度器。断点续训会恢复优化器并继续使用 checkpoint 中的学习率。如果10轮后已接近收敛，可显式降到 `5e-6`：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train_stage1.py \
  --config configs/stage1_opt2sar_pretrain.yaml \
  --manifest /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/dataset/stage1_prepared/manifest.jsonl \
  --output-dir runs/stage1_opt2sar_pretrain \
  --resume-from runs/stage1_opt2sar_pretrain/checkpoints/last.pt \
  --batch-size 16 \
  --epochs 20 \
  --lr 5e-6
```

显式传入 `--lr` 时，训练器会在恢复 optimizer 状态后覆盖其学习率；不传则保留 checkpoint 学习率。

### 13.6 nohup 后台启动双H100

`train_h100_2gpu.sh` 默认使用两张GPU、每卡batch size 64、20个epoch和Hugging Face离线缓存。启动：

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/sar-sd
mkdir -p runs/stage1_opt2sar_pretrain

nohup bash train_h100_2gpu.sh \
  > runs/stage1_opt2sar_pretrain/train.log 2>&1 &

echo $! > runs/stage1_opt2sar_pretrain/train.pid
```

查看日志：

```bash
tail -f runs/stage1_opt2sar_pretrain/train.log
```

查看进程和GPU：

```bash
cat runs/stage1_opt2sar_pretrain/train.pid
ps -fp "$(cat runs/stage1_opt2sar_pretrain/train.pid)"
watch -n 1 nvidia-smi
```

停止训练（先向torchrun主进程发送TERM）：

```bash
kill -TERM "$(cat runs/stage1_opt2sar_pretrain/train.pid)"
```

临时覆盖参数，例如每卡32、10个epoch：

```bash
BATCH_SIZE=32 EPOCHS=10 nohup bash train_h100_2gpu.sh \
  > runs/stage1_opt2sar_pretrain/train.log 2>&1 &
echo $! > runs/stage1_opt2sar_pretrain/train.pid
```

如果模型尚未缓存完整，可在前台临时启用联网模式：`OFFLINE=0 bash train_h100_2gpu.sh`。正式双卡训练建议先确认VAE和UNet已缓存，再保持默认 `OFFLINE=1`，避免两个DDP进程同时访问Hugging Face。

### 13.7 正式权重下载与1-step冒烟测试

`stage1_multidataset_smoke.yaml` 使用从头构建的小UNet，不能验证正式SD 1.5 UNet权重。`stage1_weight_download_smoke.yaml` 使用和正式训练完全相同的VAE与UNet，但只读取3个样本、使用64×64分辨率并训练1步。

先生成极小manifest：

```bash
python tools/make_tiny_manifest.py \
  --input /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/dataset/stage1_prepared/manifest.jsonl \
  --output data/weight_download_smoke/manifest.jsonl \
  --train-samples 2 \
  --val-samples 1
```

首次联网运行，必要的VAE和UNet文件会下载到 `HF_HOME`：

```bash
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export HF_HOME=/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/my_global_cache/huggingface

CUDA_VISIBLE_DEVICES=0 python train_stage1.py \
  --config configs/stage1_weight_download_smoke.yaml
```

然后用离线模式重跑。如果仍能完成1-step训练并生成 `runs/stage1_weight_download_smoke/checkpoints/last.pt`，即证明正式训练所需权重已缓存完整：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES=0 python train_stage1.py \
  --config configs/stage1_weight_download_smoke.yaml
```

本项目的正式模型只从Hub加载 `stabilityai/sd-vae-ft-mse` VAE和 `stable-diffusion-v1-5/stable-diffusion-v1-5` 下的 `unet/`；不加载tokenizer、text encoder或完整Stable Diffusion pipeline。
