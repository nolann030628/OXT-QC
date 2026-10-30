# OXT-QC

Utilities for Open-X-Tactile (OXT) dataset precheck, VLM-based semantic review, and 4D visualization.

## Repository Structure

- `precheck_metadata.py`: metadata/schema precheck; exports `precheck_report.json`, `precheck_report_result.json`, `metadata.json`, and `dict.json`
- `precheck_vlm.py`: VLM-based RGB+tactile review; exports `vlm_review.json`, `vlm_review_result.json`, and montage images
- `precheck_4dvis.py`: 4D visualization check for trajectories, coordinate frames, and rendered scene behavior
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
- In the current pipeline, tactile labels such as `matrix` and `array` are treated as equivalent 2D tactile-map formats for rendering and review purposes.
