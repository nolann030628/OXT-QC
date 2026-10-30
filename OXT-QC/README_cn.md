# OXT-QC

用于 Open-X-Tactile (OXT) 数据集的预检、VLM 语义复核与 4D 可视化。

## 仓库结构

- `precheck_metadata.py`：metadata / schema precheck，导出 `precheck_report.json`、`precheck_report_result.json`、`metadata.json` 和 `dict.json`
- `precheck_vlm.py`：基于 VLM 的 RGB+tactile review，导出 `vlm_review.json`、`vlm_review_result.json` 和 montage 图片
- `precheck_4dvis.py`：4D visualization check，用于检查轨迹、坐标系和可视化表现
- `oxt_check_output/`：默认输出目录，保存 metadata、report、VLM review 和可视化产物
- `configs/`：配置文件
- `common/`：预检和可视化脚本依赖的通用工具函数
- `src/`：辅助模块，目前包含 `report.py` 和 `vlm_checker.py`
- `tests/`：测试代码
- `FTP1_data_example/`：示例 zarr 数据
- `scripts/`：当前基本为空，可视为保留目录

## 推荐流程

1. 先运行 `precheck_metadata.py`，检查 zarr 结构、时间戳、RGB 流、tactile 四件套等硬约束。
2. 再运行 `precheck_4dvis.py`，对轨迹、坐标系、点云和时序表现做可视化检查。
3. 最后运行 `precheck_vlm.py`，对 RGB 与 tactile 的同步性、接触一致性、信号健康度和指令一致性做语义复核。

## 环境安装

```bash
pip install -r requirements.txt
```

`ReplayBuffer` 导入时会自动注册 `imagecodecs.numcodecs` 提供的 codec（不覆盖已注册的 codec），包括 JPEG、JPEG XL、JPEG 2000、JPEG-LS、PNG、WebP 等。已有环境可运行 `python -m pip install 'imagecodecs>=2025.3.30'`。没有安装时仍可读取 Zstd、LZ4 和未压缩数组，但读取 JPEG XL 等数据前必须安装该依赖并重启 Python 进程。具体 codec 是否可用取决于安装包的原生后端；此注册用于 Zarr v2 的 numcodecs 元数据，不代表支持所有自定义 codec 或 Zarr v3 codec 管线。读取时按数组保存的配置自动解码，`disk` 仅为写入默认值。

## 用法

### 1. Metadata 预检

该脚本用于检查zarr的key-value是否正确，并输出metadata用于人工检查；运行：

```bash
python precheck_metadata.py /path/to/dataset.zarr
```

输出内容在`oxt_check_output/`，其中包含：

 - `dict.json`：整个zarr的key-value对，用于核对key-value的shape是否正确；对于固定值的key（比如hand_joint_idx、tactile_type、tactile_name），还会打印固定值；图像 key（以 `_rgb`、`_depth`、`_image` 结尾，或维度不少于 4）额外输出 `compression`，记录实际存储的压缩配置，而不是推测默认值。Zarr v2 未压缩时为 `null`，v3 为压缩 codec 配置列表（未压缩时为 `[]`）。ReplayBuffer 的 `disk` 对应 Blosc + Zstd、level 5、BITSHUFFLE。
 - `metadata.json`: 最终提交需要的metadata.json脚本，需要人工填写的部分默认留空；
 - `precheck_report.json`：metadata检查结果的完整版汇总报告；
 - `precheck_report_result.json`: metadata检查结果的精简结论；

注意，如果您的数据集符合tag的情况，请在最终提交的`metadata.json`的`label`这个key中填入`umi`、`ego`、`with-supp`。推荐检查方式：(1) 打开`dict.json`，人工阅读所有的key和shape，确保没有关键的key遗漏，以及shape没有出错；(2) 打开`precheck_report_result.json`，并基于报告人工检查所有模糊项，并修正错误项。

### 2. 4D 可视化

该脚本用于对camera和wrist的坐标系和动作进行4D可视乎（可交互），用于检查坐标系是否正确；运行：

```bash
python precheck_4dvis.py -d --data_path /path/to/dataset.zarr
```

如果数据含有`umi`标签(label)，请参考`docs_关于umi数据坐标系定义的说明.pptx`；如果数据含有`ego`标签，请参考`docs_关于ego数据坐标系定义的说明.pptx`；如果都不包含，则可以不进行4dvis检查；


### 3. VLM 预检

默认使用 Ark：

```bash
# Linux
export ARK_API_KEY="your_key"
# Windows
$env:ARK_API_KEY="your_key"

python precheck_vlm.py /path/to/source_or_dataset \
  --provider ark \
  --model "doubao-seed-2-1-pro-260628"
```

使用 Azure OpenAI：

```bash
# Linux
export AZURE_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_API_KEY="your_key"
# Windows
$env:AZURE_ENDPOINT="https://your-resource.openai.azure.com/"
$env:AZURE_API_KEY="your_key"

python precheck_vlm.py /path/to/source_or_dataset \
  --provider azure \
  --model "gpt-5.4" \
  --azure-api-version "2025-03-01-preview"
```

仅测试 API 连通性：

```bash
python precheck_vlm.py /path/to/source_or_dataset \
  --provider azure \
  --model "gpt-5.4" \
  --azure-api-version "2025-03-01-preview" \
  --test-api
```

输出内容在`oxt_check_output/`，其中包含：

 - `vlm_review.json`：基于VLM的检查结果的完整版汇总报告；
 - `vlm_review_result.json`: 基于VLM的检查结果的精简结论；

推荐检查方式：打开`vlm_review_result.json`，阅读每一个打分项目中出现warning或者error的项目，阅读VLM的错误反馈，然后人工检查看错误是否存在or致命；由于VLM的检查存在不准确性，这一部分需要人工进行复核；VLM报错不代表数据实际出错，VLM报warning不代表数据不存在致命错误；VLM的反馈仅供初筛和参考。

## 说明

- `precheck_vlm.py` 会先做下采样，再把 RGB 与 tactile 片段整理成 montage，尽量降低 API 成本。
- tactile 类型里若出现 `matrix`，当前会统一归并为 `array` 处理，避免类型口径不一致。
