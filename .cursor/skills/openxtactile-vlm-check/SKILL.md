---
name: openxtactile-vlm-check
description: 对 Open-X-Tactile zarr 数据执行 owner 侧 VLM 内部检查并输出 vlm_review.json。用户提到 tactile+rgb 同步检查、断触检查、VLM 复核、Ark/doubao、openxtactile_vlm_check.py 时使用。
disable-model-invocation: true
---

# Open-X-Tactile VLM 内部检查

## 适用范围

用于仓库 `tactile-vlm-qc` 的 owner 侧语义检查，不重复做 `openxtactile_check_zarr.py` 已覆盖的硬性预检。

## 核心原则

- 先确认预检已通过，再进行 VLM 检查。
- 优先节省 token：固定使用下采样，不做全帧调用。
- 输出必须落盘为 `vlm_review.json`，并保留可视化证据图 `vlm_montage.png`。
- 检查项聚焦 4 类：
  - RGB 与 tactile 粗同步
  - 接触一致性与断触
  - 信号健康度
  - 指令与动作一致性

## 固定命令模板（Ark）

```bash
python scripts/openxtactile_vlm_check.py <zarr或父目录> \
  --provider ark \
  --api-base "https://ark.cn-beijing.volces.com/api/v3" \
  --model "doubao-seed-2-1-pro-260628" \
  -o reports/vlm_run
```

若 key 不在环境变量中，追加：

```bash
--api-key "<ARK_API_KEY>"
```

## API 先验检查

先做连通性测试，避免长任务失败：

```bash
python scripts/openxtactile_vlm_check.py <任意路径> \
  --provider ark \
  --api-base "https://ark.cn-beijing.volces.com/api/v3" \
  --model "doubao-seed-2-1-pro-260628" \
  --test-api
```

## 输出验收清单

- `vlm_review.json` 存在且可解析
- 每个数据集都有：
  - `sampling`（含 indices/uniform/events）
  - `tactile_type_raw` 与 `tactile_type_normalized`
  - `tactile_render_kind` 与 `tactile_render_range`
  - `vlm_review.final_status`
  - `artifacts.montage_png`

## 结果解释约定

- `pass`：可进入下一环节
- `warning`：建议人工抽查
- `fail`：必须人工复核并回到数据侧定位问题

## 常见问题处理

- `ModelNotOpen`：模型未开通或模型名/endpoint 不匹配，需在 Ark 控制台核对。
- `invalid_api_key`：key 无效或环境变量未设置。
- 若模型对 tactile 误判偏多，先确认渲染类型是否正确（`state` vs `array/image`）。
