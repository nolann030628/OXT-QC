# OXT-QC

用于 Open-X-Tactile (OXT) 数据集的预检、VLM 语义复核与 4D 可视化。

## 仓库结构

- `precheck_metadata.py`：metadata / schema precheck，导出 `precheck_report.json`、`precheck_report_result.json`、`metadata.json` 和 `dict.json`
- `precheck_vlm.py`：基于 VLM 的 RGB+tactile review，导出 `vlm_review.json`、`vlm_review_result.json` 和 montage 图片
- `precheck_4dvis.py`：4D visualization check，用于检查轨迹、坐标系和可视化表现
- `precheck_final.py`：输入最终 metadata 提交文件夹，对 14 主项及 HF/MS 三个子项终审，输出本地报告
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
4. 提交包整理后运行 `precheck_final.py`。它读取已导出的 dict/QC 结果、metadata 和附件，不读取 Excel，不启动新的 VLM 调用或改写数据。

## 最终提交检查

从本目录检查单个最终 metadata 包：

```powershell
python precheck_final.py ../../metadata/OXT_Submission_DreamTac
```

检查全部提交（仍向每个包内写报告）：

```powershell
python precheck_final.py ../../metadata --batch --workers 4
```

名称检查不依赖 Excel、Google Sheet、本地文件夹名称或 `metadata.name` 的显示名称。HF/MS 仓库短名必须以 `_OXT` 结尾，仓库根下的数据目录必须与当前仓库短名完全一致，例如 `owner/Demo_OXT/Demo_OXT/`。已移除 `--workbook`、`--dataset-name` 参数。

`metadata.url` 必须且只能包含一个 URL，重复 URL 或 HF/MS 各一条也不允许；`code_trigger` 必须填写空字符串 `""`。触觉 `state` 只允许三维 `(T,N,D)`，四/五维浮点空间网格应为 `matrix`，`image` 数据必须为 `uint8`。脚本直接读取 `metadata.type`：对 `gripper` 的单/双触觉区域检查局部索引 `0`/`[0,1]`，覆盖 image、matrix、state、binary 及可解析的静态索引。

每个输入目录生成英文 `precheck_final_report.json`、`precheck_final_report.md`、`precheck_final_report.html`，并额外生成中文 `precheck_final_report_cn.md`；批量父目录的 `precheck_final_summary.json/.md/.html` 也使用英文。命令行帮助、错误提示和检查消息均为英文。

中英文 Markdown 共用相同检查结果，开头直接以 `1. ...`、`2. ...` 列出 fail 问题，每个问题一句话，同类文件/字段合并；没有 fail 时分别写“No fail findings.”或“未发现 fail”。下方保留完整机器检查详情，包括 17 项检查范围、问题字段、来源、出现次数、样本证据及远端访问/覆盖记录。HTML 顶部显示英文编号列表，下方机器详情默认展开，并提供中英文 Markdown 链接。原始中文证据在英文文件中以 Unicode 转义保存，读取 JSON 后仍可还原原值，中文 Markdown 直接显示原文。

最终检查仅保留机器能核验的字段、dtype/shape、索引、附件可读性、QC 执行/评分和仓库格式。图像内容、动作语义、真实同步、物理单位、标定和坐标方向等按正确处理，不输出人工核验项；跳过 JSON 生成来源鉴定，不再按 Excel 终审备注增加字段类型或标签要求。JSON 文件的存在、可解析性、结构和已有 QC 执行结果仍会检查。

网络检查默认开启：HF 根目录、子目录或文件读取失败时尝试 hf-mirror 的相同 revision/路径；MS 使用 dataset tree API。只读取必要目录层级、README 和有界压缩包头，不全量下载大包。`--timeout`、`--repo-budget`、`--max-pages`、`--max-tasks` 控制预算；达到上限或网络/权限问题标 unknown，不判为目录不存在。

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
