# 配对 OPT-SAR 数据集接入指南

本项目不在Git仓库中分发任何数据。请先遵守各数据集许可，再使用 `tools/prepare_dataset.py` 转换为统一的图像目录和 `manifest.jsonl`。

## 1. 推荐数据集概览

### SAR-1M

- 来源：SARMAE工作构建，2025年预印本、CVPR 2026。
- 规模：超过130万张SAR图像；本地整包中的 `paired.json` 实际列出731,080对OPT-SAR，另有399,429张非配对SAR。
- 本地文件：`D:\BaiduNetdiskDownload\SAR-Data\SAR-1M_DATA.zip`。
- 用途：Stage 1大规模预训练主体。
- 注意：整包约76.6GB；元数据并不保证每个样本都有可靠GSD、传感器和极化字段，未知字段必须保留为 `unknown`。

抽取1000对做代码验证：

```powershell
python tools\prepare_dataset.py sar1m-zip `
  --source "D:\BaiduNetdiskDownload\SAR-Data\SAR-1M_DATA.zip" `
  --output "D:\opt2sar_data\sar1m_1k" `
  --limit 1000 `
  --opt-sensor unknown --sar-sensor unknown `
  --opt-gsd 10 --sar-gsd 10 --polarization unknown
```

这里的10m只是元数据缺失时的占位默认值，不能作为论文中的真实传感器参数。正式使用前应根据SAR-1M来源表进一步补全元数据。

### SEN1-2

- 发布：TUM/DLR，2018年。
- 规模：282,384对，256×256，覆盖全球和四季。
- 传感器：Sentinel-2 RGB与Sentinel-1 VV-IW GRD。
- GSD：数据导出到10m网格。
- 许可：CC BY 4.0。
- 官方地址：https://mediatum.ub.tum.de/1436631

下载极少量官方样本：

```powershell
python tools\prepare_multidataset_smoke.py --output data\multidataset_smoke --count 2
```

### OSDataset 2.0

本地Word说明确认：该数据由中科院空天信息创新研究院相关实验室发布，SAR来自GF-3 C波段Spotlight模式，光学来自Google Earth，空间分辨率1m。包含2673对512×512非重叠图块和10692对256×256非重叠图块，均为8位图像，仅限科研、禁止商业用途。

它不是QXS-SAROPT。二者虽然都使用GF-3与Google Earth，但规模、发布团队和覆盖范围不同。

```powershell
python tools\prepare_dataset.py osdataset `
  --source "D:\BaiduNetdiskDownload\SAR-Data\OSDataset2.0\OSDataset2.0\Patch-level Subset" `
  --output "D:\opt2sar_data\osdataset" `
  --opt-sensor Google-Earth --sar-sensor GF-3 `
  --opt-gsd 1 --sar-gsd 1 --polarization unknown
```

### MOS-Ship

本地目录已经是 `_rgb.png` 与 `_sar.png` 成对格式，可用于流程验证或舰船场景补充。它更偏目标域数据，建议不要让其在通用预训练中占比过高。

```powershell
python tools\prepare_dataset.py mos-ship `
  --source "D:\BaiduNetdiskDownload\SAR-Data\MOS-Ship\MOS-Ship" `
  --output "D:\opt2sar_data\mos_ship" `
  --opt-sensor unknown --sar-sensor unknown `
  --opt-gsd 1 --sar-gsd 1 --polarization unknown
```

### QXS-SAROPT（QXSLAB）

- 发布：论文预印本2021年。
- 规模：20,000对，256×256，约1m像素间距。
- SAR：GF-3；OPT：Google Earth。
- 地区：青岛、上海、圣迭戈三个港口城市。
- 官方仓库：https://github.com/yaoxu008/QXS-SAROPT
- 仓库中的下载入口需要按作者页面完成申请/下载，数据本体不在Git仓库中。

解压后按实际目录名执行通用适配器：

```powershell
python tools\prepare_dataset.py paired-dirs `
  --source "D:\datasets\QXS-SAROPT" `
  --output "D:\opt2sar_data\qxs_saropt" `
  --dataset QXS-SAROPT --opt-dir optical --sar-dir sar `
  --opt-sensor Google-Earth --sar-sensor GF-3 `
  --opt-gsd 1 --sar-gsd 1 --polarization unknown
```

如果文件分别以 `opt_0001` 和 `sar_0001` 命名，再增加 `--opt-prefix opt_ --sar-prefix sar_`。

### WHU-OPT-SAR

- 发布方：武汉大学。
- 原始规模：100对大幅影像，每幅约5556×3704。
- 覆盖：湖北省约50,000平方千米。
- OPT：GF-1，原始约2m；SAR：GF-3，约5m。
- 对齐：OPT重采样到5m并与SAR亚像素配准。
- 常见切片版本：非重叠512×512后约7000对，但这属于派生设置，使用时应说明切片规则。

```powershell
python tools\prepare_dataset.py paired-dirs `
  --source "D:\datasets\WHU-OPT-SAR" `
  --output "D:\opt2sar_data\whu_opt_sar" `
  --dataset WHU-OPT-SAR --opt-dir optical --sar-dir sar `
  --opt-sensor GF-1 --sar-sensor GF-3 `
  --opt-gsd 5 --sar-gsd 5 --polarization unknown
```

这里记录的是对齐后网格GSD；如果保留GF-1原始2m数据，应在预处理阶段先决定共同网格，不能仅靠网络resize。

### SOMA-1M

- 发布方：武汉大学遥感信息工程学院等，2026年。
- 规模：1,300,954对像素级对齐图像，512×512。
- 地理范围：全球1466个采样点，12类地表场景。
- 分辨率：0.5–10m。
- 传感器：Sentinel-1、PIESAT-1、Capella Space和Google Earth等。
- 官方仓库：https://github.com/PeihaoWu/SOMA-1M
- 当前官方仓库提供test子集的Google Drive和百度网盘链接，更多子集仍在准备发布。

SOMA-1M不能为整个数据集填写一个固定GSD。转换manifest时必须从官方索引逐样本写入 `opt_gsd`、`sar_gsd`、传感器和场景ID。若下载版本只提供按传感器/GSD划分的目录，可分别运行通用适配器，再合并manifest。

```powershell
python tools\prepare_dataset.py paired-dirs `
  --source "D:\datasets\SOMA-1M\sentinel1_10m" `
  --output "D:\opt2sar_data\soma_s1_10m" `
  --dataset SOMA-1M --opt-dir optical --sar-dir sar `
  --opt-sensor Google-Earth --sar-sensor Sentinel-1 `
  --opt-gsd 10 --sar-gsd 10 --polarization unknown
```

### SAR2Opt 与 3MOS

- SAR2Opt：2076对600×600、1m TerraSAR-X/Google Earth图像；公开镜像可由smoke脚本抽取。
- 3MOS：约11.3万对、五种SAR卫星、3.5–12.5m、八类场景；官方数据入口主要为百度网盘。

两者都可以用 `paired-dirs` 适配器，3MOS应按卫星分别生成manifest，不能全部标成同一传感器和GSD。

## 2. 合并manifest

每个数据集先独立转换和检查，再合并：

```powershell
python tools\merge_manifests.py `
  D:\opt2sar_data\sar1m_1k\manifest.jsonl `
  D:\opt2sar_data\osdataset\manifest.jsonl `
  D:\opt2sar_data\qxs_saropt\manifest.jsonl `
  --output D:\opt2sar_data\stage1_all\manifest.jsonl
```

建议正式训练前统计每个数据集、传感器、GSD和极化的样本数，并人工打开随机配对图确认方向、配准和动态范围。

## 3. 不能混入配对训练的数据

SAR-JEPA、SARATR-X、SSDD、SAR-Ship等SAR单模态数据不能直接作为OPT→SAR配对扩散训练样本。它们可以用于SAR VAE适配或SAR先验学习，但必须与Stage 1-B的配对训练数据分开管理。
