# OXT-QC

Utilities for Open-X-Tactile (OXT) dataset precheck, VLM-based semantic review, and 4D visualization.

## Repository Structure

- `precheck_metadata.py`: metadata/schema precheck; exports `precheck_report.json`, `precheck_report_result.json`, `metadata.json`, and `dict.json`
- `precheck_vlm.py`: VLM-based RGB+tactile review; exports `vlm_review.json`, `vlm_review_result.json`, and montage images
- `precheck_4dvis.py`: 4D visualization check for trajectories, coordinate frames, and rendered scene behavior
- `precheck_final.py`: final check of a metadata submission folder against 14 main checks and three HF/MS subchecks; writes local reports
- `oxt_check_output/`: default output directory for metadata, reports, VLM review results, and generated images
- `configs/`: configuration files
- `common/`: shared helper utilities used by the precheck and visualization scripts
- `src/`: auxiliary modules, currently including `report.py` and `vlm_checker.py`
- `tests/`: tests
- `FTP1_data_example/`: example zarr datasets
- `scripts/`: currently mostly empty; effectively a reserved directory

## Recommended Workflow

1. Run `precheck_metadata.py` first to check hard constraints such as zarr structure, timestamps, RGB streams, and tactile four-piece sets.
2. Run `precheck_4dvis.py` next to visually inspect trajectories, coordinate systems, point clouds, and temporal behavior.
3. Run `precheck_vlm.py` last to perform semantic review of RGB-tactile synchronization, contact consistency, signal health, and instruction consistency.
4. After preparing the submission folder, run `precheck_final.py`. It reads exported dict/QC results, the Contributor List, and submission assets without starting new VLM calls or modifying dataset contents.

## Final Submission Check

From this directory, check a single final metadata submission. The Contributor List is located in ancestor directories by default:

```powershell
python precheck_final.py ../../metadata/OXT_Submission_DreamTac
```

Check all submissions, writing reports inside each submission folder:

```powershell
python precheck_final.py ../../metadata --batch --workers 4
```

You can also specify `--workbook "../../Open-X-Tactile Contributor List.xlsx"` explicitly. For a single submission, `--dataset-name` selects a registered name that exists in the workbook. Alias recognition only identifies the workbook row; naming compliance still uses exact comparisons.

Each input folder receives English `precheck_final_report.json`, `precheck_final_report.md`, and `precheck_final_report.html`, plus Chinese `precheck_final_report_cn.md`. Batch runs also write English `precheck_final_summary.json`, `precheck_final_summary.md`, and `precheck_final_summary.html` in the parent directory. Command-line help, error messages, and check messages are in English.

Both Markdown versions use the same check results. English reports start with `# <dataset_name> Final Data Review Results`, followed by `## Required Changes` and a numbered list of fail findings (`1. ...`, `2. ...`). Chinese reports use `# <dataset_name>数据终审结果` and `## 需要修改的地方`. The title uses `metadata.name`, falling back to the dataset name in the submission folder when unavailable. Each issue is one sentence, with related files/fields grouped together. When there are no fail findings, the English version says "No fail findings." and the Chinese version says "未发现 fail。" Full machine check details follow, including coverage of all 17 checks, affected fields, sources, occurrence counts, sample evidence, and remote access/coverage records. HTML shows the same English numbered list at the top, expands machine details by default, and links to both Markdown versions. Original Chinese evidence is preserved as Unicode escapes in English files; JSON parsing restores the original values, while Chinese Markdown displays the original text directly.

The final check covers machine-verifiable fields, dtype/shape, indices, asset readability, QC execution/scores, and repository layout. Image content, action semantics, actual synchronization, physical units, calibration, and coordinate directions are treated as correct for this check, and no manual verification items are emitted. State/matrix semantics that cannot be determined from rank alone are not reported as separate issues; field types explicitly specified in the final review are still checked against those requirements.

Network checks are enabled by default. If reading an HF root directory, subdirectory, or file fails, the tool tries hf-mirror at the same revision/path. MS uses the dataset tree API. The tool reads only necessary directory levels, README text, and bounded archive headers, without downloading full large archives. `--timeout`, `--repo-budget`, `--max-pages`, and `--max-tasks` control access budgets. Reaching a limit or encountering a network/permission issue results in `unknown`, rather than a finding that a directory is missing.

## Installation

```bash
pip install -r requirements.txt
```

Importing `ReplayBuffer` automatically registers the codecs provided by `imagecodecs.numcodecs` without replacing existing registrations, including JPEG, JPEG XL, JPEG 2000, JPEG-LS, PNG, and WebP. For an existing environment, run `python -m pip install 'imagecodecs>=2025.3.30'`. Without it, Zstd, LZ4, and uncompressed arrays still work; install the dependency and restart Python before loading JPEG XL data. Individual codecs require their native backend in the installed package. This registration covers Zarr v2 numcodecs metadata, not every custom codec or Zarr v3 codec pipeline. Reading uses the stored configuration automatically; `disk` is only the write default.

## Usage

### 1. Metadata Precheck

This script validates the zarr key/value structure and generates metadata artifacts for manual review.

```bash
python precheck_metadata.py /path/to/dataset.zarr
```

All outputs are written to `oxt_check_output/` by default:

- `dict.json`: a full key summary for the current zarr, including shape and dtype for every key; for constant-like keys such as `hand_joints_idx`, `tactile_type`, and tactile area/sensor identifiers, the constant value is also recorded. Image keys (ending in `_rgb`, `_depth`, or `_image`, or having at least four dimensions) also include `compression`, read from the actual stored configuration. Uncompressed Zarr v2 arrays use `null`; v3 arrays use a list of compression codec configurations (`[]` when uncompressed). ReplayBuffer's `disk` preset means Blosc + Zstd, level 5, BITSHUFFLE.
- `metadata.json`: the metadata file intended for final submission; human-authored fields are left empty by default when they cannot be inferred from the dataset
- `precheck_report.json`: the full metadata precheck report
- `precheck_report_result.json`: the compact metadata precheck conclusion

If your dataset should be tagged, please fill the `label` field in the final `metadata.json` manually, for example with `umi`, `ego`, or `with-supp`.

Recommended review method:
1. Open `dict.json` and manually review all keys and shapes to ensure no critical key is missing and no shape is incorrect.
2. Open `precheck_report_result.json`, manually inspect every ambiguous item mentioned in the report, and correct any actual errors.

### 2. 4D Visualization Check

This script performs interactive 4D visualization for camera/wrist coordinate systems and motion behavior, mainly for visual validation.

```bash
python precheck_4dvis.py -d --data_path /path/to/dataset.zarr
```

If the dataset carries the `umi` label, refer to `docs_Illustration about UMI Data Frame.pptx`.
If the dataset carries the `ego` label, refer to `docs_Illustration about Ego Data Frame.pptx`.
If neither label applies, this 4D visualization check may be skipped.

### 3. VLM Precheck

Default Ark example:

```bash
# Linux
export ARK_API_KEY="your_key"

# Windows PowerShell
$env:ARK_API_KEY="your_key"

python precheck_vlm.py /path/to/source_or_dataset \
  --provider ark \
  --model "doubao-seed-2-1-pro-260628"
```

Azure OpenAI example:

```bash
# Linux
export AZURE_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_API_KEY="your_key"

# Windows PowerShell
$env:AZURE_ENDPOINT="https://your-resource.openai.azure.com/"
$env:AZURE_API_KEY="your_key"

python precheck_vlm.py /path/to/source_or_dataset \
  --provider azure \
  --model "gpt-5.4" \
  --azure-api-version "2025-03-01-preview"
```

API connectivity test only:

```bash
python precheck_vlm.py /path/to/source_or_dataset \
  --provider azure \
  --model "gpt-5.4" \
  --azure-api-version "2025-03-01-preview" \
  --test-api
```

All outputs are written to `oxt_check_output/` by default:

- `vlm_review.json`: the full VLM review log, including prompts, raw responses, per-page reviews, and generated image artifacts
- `vlm_review_result.json`: the compact VLM review result organized by review aspect

Recommended review method:
Open `vlm_review_result.json` and examine every item that appears under `warning` or `error` in each review aspect. Read the VLM feedback, then manually verify whether the issue truly exists and whether it is fatal. Because VLM review is not fully reliable, this part must be manually checked. A VLM `error` does not necessarily mean the dataset is actually wrong, and a VLM `warning` does not necessarily mean the dataset has no fatal issue. VLM feedback should be used only for initial screening and reference.

## Notes

- `precheck_vlm.py` first downsamples frames, then packs RGB+tactile snippets into montage images to reduce API cost.
- If tactile types include `matrix`, the current pipeline normalizes them to `array` for processing to keep type handling consistent.
