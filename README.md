# tactile-vlm-qc

用于 Open-X-Tactile 数据质量检查与 VLM 语义复核的工具仓库。

## 功能概览

- `openxtactile_check_zarr.py`：数据预检（结构完整性、时间戳、四件套、基础元信息）
- `scripts/openxtactile_vlm_check.py`：VLM 内部检查（RGB 与 tactile 粗同步、接触一致性、信号健康）
- 结果输出为结构化 JSON，便于 CI 与人工复核

## 当前流程（推荐）

1. 先跑预检脚本（硬性质量门槛）
2. 再跑 VLM 检查脚本（语义层与同步层复核）
3. 综合 `precheck_report.json` 与 `vlm_review.json` 判定 pass/warning/fail

## 目录说明

- `scripts/`：可执行入口脚本
- `configs/`：配置文件
- `reports/`：运行产物（JSON、可视化图片）
- `data/`：本地测试数据目录
- `tests/`：测试代码

## 环境准备

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install zarr numpy pillow matplotlib imageio "volcengine-python-sdk[ark]"
```

## 预检示例

```bash
python openxtactile_check_zarr.py /path/to/source_or_dataset \
  -o reports/precheck_output
```

## VLM 检查示例（Ark）

```bash
export ARK_API_KEY="你的 key"
python scripts/openxtactile_vlm_check.py /path/to/source_or_dataset \
  --provider ark \
  --api-base "https://ark.cn-beijing.volces.com/api/v3" \
  --model "doubao-seed-2-1-pro-260628" \
  -o reports/vlm_output
```

## 输出文件

- 预检：`precheck_report.json`、`metadata.json`
- VLM：`vlm_review.json`、每个数据集的 `vlm_montage.png`

## 说明

- VLM 脚本默认做下采样（均匀抽样 + 事件抽样）以减少 API 成本。
- `matrix` 类型已在脚本中归一到 `array`，避免类型口径不一致。

# OXT-QC
