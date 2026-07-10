# OXT-QC

Open-X-Tactile (OXT) dataset quality check toolkit.

Open-X-Tactile (OXT) 数据集质量检查工具集。

## Overview | 项目概览

- Metadata and zarr precheck (`precheck_metadata.py`)
- VLM-based RGB+tactile semantic review (`precheck_vlm.py`)
- 4D visualization check for pose/alignment (`precheck_4dvis.py`)

- metadata 与 zarr 结构预检（`precheck_metadata.py`）
- 基于 VLM 的 RGB+tactile 语义复核（`precheck_vlm.py`）
- 坐标系/轨迹相关 4D 可视化检查（`precheck_4dvis.py`）

## Quick Start | 快速开始

```bash
pip install -r requirements.txt
```

### 1) Metadata precheck | metadata 预检

```bash
python precheck_metadata.py /path/to/zarr_or_parent_dir -o oxt_check_output
```

### 2) VLM review | VLM 复核

```bash
python precheck_vlm.py /path/to/zarr_or_parent_dir -o oxt_check_output
```

### 3) 4D visualization (optional/recommended for pose data) | 4D 可视化（可选，含 pose 数据时推荐）

```bash
python precheck_4dvis.py -d --data_path /path/to/one_dataset.zarr
```

## Outputs | 主要输出

- `oxt_check_output/precheck_report.json`
- `oxt_check_output/precheck_report_result.json`
- `oxt_check_output/metadata.json`
- `oxt_check_output/dict.json`
- `oxt_check_output/vlm_review.json`
- `oxt_check_output/vlm_review_result.json`

## Full Docs | 详细文档

- Chinese: `README_cn.md`
- English: `README_en.md`
