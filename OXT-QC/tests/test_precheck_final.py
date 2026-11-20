"""Final review regression tests; all remote interfaces are mocked."""

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest
from openpyxl import Workbook
from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "precheck_final.py"
spec = importlib.util.spec_from_file_location("oxt_precheck_final_tested", MODULE_PATH)
final = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = final
spec.loader.exec_module(final)


def info(shape, dtype="float32", value=None):
    result = {"shape": shape, "dtype": dtype}
    if value is not None:
        result["constant_value"] = value
    return result


def table(T=3):
    return {
        "timestamps": info([T], "float64"),
        "camera_main_rgb": info([T, 8, 8, 3], "uint8"),
        "right_hand_joints": info([T, 1]),
        "right_hand_joints_idx": info([T, 1], "int64", [28]),
        "right_tactile_data_gripper": info([T, 1, 8, 8, 3], "uint8"),
        "right_tactile_area_gripper": info([T, 1], "int64", [0]),
        "right_tactile_sensor_gripper": info([T], "<U8", "Pad"),
        "right_tactile_type_gripper": info([T], "<U8", "image"),
        "sub_task_instruction": info([T], "<U16"),
        "meta/episode_ends": info([1], "int64", [T]),
    }


def metadata(name="Demo"):
    return {"name": name, "url": [f"https://huggingface.co/datasets/owner/{name}_OXT"],
            "version": "1.0.0", "description": "A tactile dataset", "type": "gripper",
            "sensor_image": ["Pad"], "sensor_array": [], "sensor_state": [],
            "task": 1, "trajectory": 1, "frame": 3, "label": [], "contact": ["review@example.org"],
            "reference": {"paper": "", "authors": "Reviewer", "website": ""}, "code_trigger": "FTP_1_Source"}


def registry(tmp_path, rows):
    book = Workbook()
    sheet = book.active
    sheet.title = "WeChat"
    sheet.append(["编号", "数据集名称", "联系人", "时间", "初审", "复审", "终审"])
    for i, (name, note) in enumerate(rows, 1):
        sheet.append([i, name, None, None, None, None, note])
    path = tmp_path / "list.xlsx"
    book.save(path)
    return final.Registry(path)


def empty_directives():
    return {"field_types": {}, "scalar_fields": {}, "review": ""}


def codes(collector):
    return {f["code"] for f in collector.findings}


def entry(path, kind="file", size=100):
    return {"path": path, "kind": kind, "size": size}


def test_qc_wrapper_is_counted_once_and_richer_constants_are_used():
    raw = {"one.zarr": {"keys": {"timestamps": info([3])}, "dict_export": table()}}
    tables = list(final.iter_tables(raw))
    assert len(tables) == 1
    assert tables[0][1]["right_tactile_type_gripper"]["constant_value"] == "image"


def test_registry_preserves_official_spelling_and_rejects_ambiguity(tmp_path):
    reg = registry(tmp_path, [("AB", ""), ("A-B", "")])
    record, reason = reg.resolve(Path("OXT_Submission_a_b"), {"name": "a b"})
    assert record is None
    assert set(reason["candidates"]) == {"AB", "A-B"}
    record, _ = reg.resolve(Path("OXT_Submission_A-B"), {"name": "A-B"})
    assert record["name"] == "A-B"


def test_legacy_separators_identify_but_do_not_pass_names(tmp_path):
    reg = registry(tmp_path, [("TactileLab-InHandRotation", "")])
    folder = Path("OXT_Submission_TactileLab_InHandRotation")
    meta = metadata("TactileLab_InHandRotation")
    record, resolution = reg.resolve(folder, meta)
    assert record["name"] == "TactileLab-InHandRotation"
    c = final.Collector()
    final.validate_metadata(folder, meta, record, resolution, c)
    assert {"metadata_name", "submission_name"}.issubset(codes(c))


@pytest.mark.parametrize("note,key,expected", [
    ("right_tactile_data_finger的tactile type应该是matrix而不是state", "right_tactile_data_finger", "matrix"),
    ("right_tactile_data_temperature的shape是(T,N,D)，但是type是matrix而不是state", "right_tactile_data_temperature", "state"),
    ("left_tactile_data_displacement的shape是(T,N,H,W)，但是type是state而不是matrix", "left_tactile_data_displacement", "matrix"),
])
def test_terminal_types_next_to_chinese_and_reverse_direction(note, key, expected):
    assert final.directives({"review": note})["field_types"][key] == expected


def test_terminal_scalar_axis_and_functional_area_consolidation(tmp_path):
    t = table()
    t["right_tactile_data_force"] = info([3, 2])
    t["right_tactile_area_force"] = info([3, 2], "int64", [0, 1])
    t["right_tactile_sensor_force"] = info([3], "<U8", "Force")
    t["right_tactile_type_force"] = info([3], "<U8", "state")
    t["right_tactile_area_gripper"] = info([3, 1], "int64", [2])
    directive = final.directives({"review": "right_tactile_data_force的shape应该是(T,2,1)"})
    c = final.Collector()
    final.check_arrays(tmp_path, metadata(), {"one.zarr": t}, directive, c)
    assert {"scalar_axis_missing", "terminal_scalar_shape", "gripper_area"}.issubset(codes(c))
    t["right_tactile_data_force"] = info([3, 24, 3])
    t["right_tactile_area_force"] = info([3, 24], "int64", [0] * 24)
    c = final.Collector()
    final.check_arrays(tmp_path, metadata(), {"one.zarr": t}, empty_directives(), c)
    assert "functional_area_not_consolidated" in codes(c)


def test_supplementary_no_area_and_static_slots_do_not_false_fail(tmp_path):
    t = table()
    t.pop("right_hand_joints_idx")
    t["meta/action_slot_idx__right_hand_joints"] = info([1], "int32", [44])
    t.pop("right_tactile_area_gripper")
    t["meta/tactile_slot_idx__right_tactile_data_gripper"] = info([1], "int32", [0])
    t.update({"supplementary_tactile_data_force": info([3, 2, 1]),
              "supplementary_tactile_sensor_force": info([3], "<U8", "Force"),
              "supplementary_tactile_type_force": info([3], "<U8", "state")})
    c = final.Collector()
    final.check_arrays(tmp_path, metadata(), {"one.zarr": t}, empty_directives(), c)
    assert not any(f["status"] == "fail" for f in c.findings)


def test_extended_pose_is_not_standard_pose_and_camera_is_not_action(tmp_path):
    t = table()
    t["left_hand_pose"] = info([3, 21, 3])
    t["right_wrist_pose"] = info([3, 7])
    c = final.Collector()
    final.check_arrays(tmp_path, metadata(), {"one.zarr": t}, empty_directives(), c)
    assert any(f["code"] == "standard_pose_shape" and f["status"] == "fail" for f in c.findings)
    assert not any(f["code"] == "extended_pose" or f["status"] == "manual" for f in c.findings)
    assert final.action_keys({"camera_ego_pose": info([3, 6]), "right_hand_joints_idx": info([3, 1])}) == []


def test_conflicting_legacy_and_static_slots_are_detected(tmp_path):
    t = table()
    t["meta/action_slot_idx__right_hand_joints"] = info([1], "int32", [45])
    t["meta/tactile_slot_idx__right_tactile_data_gripper"] = info([1], "int32", [1])
    c = final.Collector()
    final.check_arrays(tmp_path, metadata(), {"one.zarr": t}, empty_directives(), c)
    assert {"legacy_static_idx_conflict", "legacy_static_area_conflict"}.issubset(codes(c))


def test_two_pads_in_separate_groups_keep_zero_and_one(tmp_path):
    t = table()
    for key in list(t):
        if "tactile_" in key:
            del t[key]
    for name, area in (("left_pad", 0), ("right_pad", 1)):
        t[f"right_tactile_data_{name}"] = info([3, 1, 8, 8, 3], "uint8")
        t[f"right_tactile_area_{name}"] = info([3, 1], "int64", [area])
        t[f"right_tactile_sensor_{name}"] = info([3], "<U8", "Pad")
        t[f"right_tactile_type_{name}"] = info([3], "<U8", "image")
    c = final.Collector()
    final.check_arrays(tmp_path, metadata(), {"one.zarr": t}, empty_directives(), c)
    assert "gripper_area" not in codes(c)
    t["right_tactile_area_left_pad"]["constant_value"] = [2]
    t["right_tactile_area_right_pad"]["constant_value"] = [3]
    c = final.Collector()
    final.check_arrays(tmp_path, metadata(), {"one.zarr": t}, empty_directives(), c)
    assert "gripper_area" in codes(c)


def test_chinese_numeric_keys_contacts_and_label_white_list(tmp_path):
    meta = metadata()
    meta.update(contact=["Name, mail@example.org"], label=["ALOHA"], institution={"中文": 1})
    c = final.Collector()
    final.validate_metadata(tmp_path, meta, None, {}, c)
    assert {"contact_email", "labels", "metadata_chinese"}.issubset(codes(c))


def test_layout_formats_root_readme_and_unknown_extensions():
    c = final.Collector()
    assert final.check_layout([entry("Demo/a.zip"), entry("Demo/b.7z")], "Demo", {}, c) == "A"
    assert not c.findings
    c = final.Collector()
    assert final.check_layout([entry("Demo/task", "directory")], "Demo", {"Demo/task": final.Tree([entry("Demo/task/data.rar")])}, c) == "B"
    assert not c.findings
    c = final.Collector()
    final.check_layout([entry("Demo/task", "directory"), entry("Demo/README.md")], "Demo", {"Demo/task": final.Tree([entry("Demo/task/raw.zarr", "directory")])}, c)
    assert {"mixed_layout", "nested_task"}.issubset(codes(c))
    assert final.archive_kind("data.unusual", b"7z\xbc\xaf\x27\x1c") == "archive"
    assert final.archive_kind("data.unusual") == "unknown"
    assert final.archive_kind("data.7z.001") == "archive"
    assert final.archive_kind("data.zip", b"<!doctype html>network gateway") == "unknown"


def test_partial_empty_list_is_unknown_not_missing_or_pass():
    c = final.Collector()
    final.check_layout([], "Demo", {}, c, complete=False)
    assert codes(c) == {"tree_partial"}
    assert next(r for r in c.summary() if r["rule"] == "10.2")["status"] == "unknown"


class MockHF(final.RemoteClient):
    def __init__(self, options):
        super().__init__(options)
        self.calls = []

    def request(self, url, deadline, **kwargs):
        self.calls.append(url)
        if url.startswith(self.options.hf_endpoint):
            raise final.RemoteError("network timeout")
        if "/tree/" not in url:
            return {"sha": "a" * 40}, {}
        if "cursor=two" in url:
            return [{"path": "README.md", "type": "file", "size": 20}], {}
        return [{"path": "Demo", "type": "directory", "size": 0}], {"link": '<' + self.options.mirror_endpoint + '/api/datasets/o/Demo_OXT/tree/' + 'a' * 40 + '?cursor=two>; rel="next"'}


def test_hf_fallback_pagination_and_revision_are_preserved():
    options = final.Options(hf_endpoint="https://official.example", mirror_endpoint="https://mirror.example")
    client = MockHF(options)
    source = final.RemoteSource("hf", "o/Demo_OXT")
    result = client.root(source)
    assert result["endpoint"] == options.mirror_endpoint
    assert result["root"]["pages"] == 2
    assert result["root"]["complete"]
    assert result["pinned"] and result["revision"] == "a" * 40
    assert any(c.startswith(options.hf_endpoint) for c in client.calls)


def test_known_official_commit_is_not_lost_during_root_fallback():
    class Client(final.RemoteClient):
        def request(self, url, deadline, **kwargs):
            if url.startswith(self.options.hf_endpoint):
                if "/tree/" in url:
                    raise final.RemoteError("HTTP 503")
                return {"sha": "b" * 40}, {}
            if "/tree/" not in url:
                assert "/revision/" + "b" * 40 in url
                return {"sha": "b" * 40}, {}
            assert "/tree/" + "b" * 40 in url
            return [], {}
    client = Client(final.Options(hf_endpoint="https://official.example", mirror_endpoint="https://mirror.example"))
    result = client.root(final.RemoteSource("hf", "owner/Demo_OXT"))
    assert result["revision"] == result["official_sha"] == "b" * 40


def test_ms_total_count_and_missing_files_are_not_silently_complete():
    class MockMS(final.RemoteClient):
        def request(self, url, deadline, **kwargs):
            return {"Code": 200, "Data": {"Files": [{"Path": "Demo", "Type": "tree", "Size": 0}], "TotalCount": 2}}, {}
    tree = MockMS(final.Options()).list_ms(final.RemoteSource("ms", "o/Demo"), "", time.monotonic() + 10)
    assert not tree.complete
    assert len(tree.entries) == 1


def test_child_hf_failure_also_uses_mirror():
    class Client:
        def root(self, source):
            return {"provider": "hf", "repo": source.repo, "requested_revision": None, "revision": "a" * 40,
                    "pinned": True, "endpoint": "https://huggingface.co", "attempts": [], "root": {"complete": True, "entries": [entry("Demo", "directory")], "pages": 1}}

        def list_hf(self, endpoint, source, revision, parent, deadline):
            if endpoint == "https://huggingface.co":
                raise final.RemoteError("HTTP 503")
            assert revision == "a" * 40 and parent == "Demo"
            return final.Tree([entry("Demo/one.zip")])

    c = final.Collector()
    result = final.check_remote(metadata(), "Demo", final.Options(archive_samples=0), Client(), c)
    assert result[0]["layout"] == "A"
    assert "mirror_subtree_used" in codes(c)
    assert "dataset_tree_unavailable" not in codes(c)


def test_batch_writes_reports_for_bad_input_and_does_not_modify_sources(tmp_path, monkeypatch):
    reg = registry(tmp_path, [("Demo", "")])
    folder = tmp_path / "OXT_Submission_Demo"
    folder.mkdir()
    qdir = folder / "oxt_check_output"
    qdir.mkdir()
    meta = metadata()
    (folder / "metadata.json").write_text(json.dumps(meta))
    (qdir / "dict.json").write_text(json.dumps({"one.zarr": table()}))
    (qdir / "precheck_report_result.json").write_text(json.dumps({"datasets": [], "overall_status": "pass"}))
    (qdir / "vlm_review_result.json").write_text(json.dumps({"status": "not_run", "reason": "no API"}))
    for name in ("hand_joints_idx.png", "tactile_func_area_idx.png", "institution_Lab.png", "sensor_Pad.png"):
        Image.new("RGB", (8, 8), "white").save(folder / name)
    (folder / "video.mp4").write_bytes(b"test fixture")
    monkeypatch.setattr(final, "probe_video", lambda path: {"width": 224, "height": 224})
    before = (folder / "metadata.json").read_bytes()
    bad = tmp_path / "OXT_Submission_Broken"
    bad.mkdir()
    (bad / "metadata.json").write_text("invalid JSON")
    assert final.main([str(tmp_path), "--batch", "--workbook", str(reg.path), "--offline", "--workers", "2"]) == 1
    for target in (folder, bad):
        report = json.loads((target / "precheck_final_report.json").read_text(encoding="utf-8"))
        assert len(report["rules"]) == 17
        assert (target / "precheck_final_report.html").exists()
        assert (target / "precheck_final_report.md").exists()
        assert report["review_scope"] == "machine_only"
        assert all(f["status"] != "manual" for f in report["findings"])
        lines = (target / "precheck_final_report.md").read_text(encoding="utf-8").splitlines()
        overview = [f"{i}. {final.html.escape(text)}" for i, text in enumerate(report["failure_points"], 1)]
        assert lines[:len(overview)] == overview
        assert "## Machine check details" in lines
        assert "## Check coverage" in lines and "## Findings and evidence" in lines
        assert (target / "precheck_final_report_cn.md").is_file()
        assert "## 机器检查详情" in (target / "precheck_final_report_cn.md").read_text(encoding="utf-8")
    assert (folder / "metadata.json").read_bytes() == before
    result = json.loads((folder / "precheck_final_report.json").read_text(encoding="utf-8"))
    assert "vlm_not_run" in {r["code"] for r in result["findings"]}
    assert next(r for r in result["rules"] if r["rule"] == "10.2")["status"] == "unknown"
    assert json.loads((tmp_path / "precheck_final_summary.json").read_text(encoding="utf-8"))["dataset_count"] == 2


def test_html_escapes_untrusted_metadata_and_report_json_is_standard():
    c = final.Collector()
    report = {"dataset": "<script>bad</script>", "canonical_name": None, "generated_at": "now", "overall_status": "fail",
              "rules": c.summary(), "findings": [], "limits": []}
    assert "<script>bad</script>" not in final.report_html(report)
    assert "&lt;script&gt;bad&lt;/script&gt;" in final.report_html(report)
    assert json.dumps(final.json_safe({"v": float("nan")}), allow_nan=False) == '{"v": "nan"}'


def test_empty_metadata_object_never_passes(tmp_path, monkeypatch):
    reg = registry(tmp_path, [("Demo", "")])
    folder = tmp_path / "OXT_Submission_Demo"
    folder.mkdir()
    (folder / "metadata.json").write_text("{}")
    monkeypatch.setattr(final, "probe_video", lambda path: {"error": "missing"})
    report = final.review_dataset(folder, reg, final.Options(offline=True), final.RemoteClient(final.Options(offline=True)))
    assert report["overall_status"] == "fail"
    assert "missing_fields" in {f["code"] for f in report["findings"]}


def test_machine_report_groups_failures_and_drops_legacy_semantic_checks():
    c = final.Collector()
    for key in ("right_tactile_data_pad", "left_tactile_data_pad", "right_tactile_data_pad"):
        c.add("01", "fail", "image_not_uint8", "A verbose explanation", key=key)
    c.add("10.1", "fail", "repo_suffix", "A verbose explanation", evidence={"expected_suffix": "Demo_OXT"})
    c.add("06", "manual", "annotation_content", "Check image contents")
    c.add("11", "unknown", "full_report_omitted", "Verify historical provenance")
    c.add("02", "warning", "matrix_vector", "Interpret physical axes")
    c.add("10.2", "unknown", "offline", "Network check unavailable")
    c.add("11", "warning", "vlm_reported_warning", "VLM warning", evidence={"reason": ["基线记录不一致，需人工核查原始数据"]})
    report = {"dataset": "Demo", "canonical_name": "Demo", "generated_at": "now",
              "rules": [{"rule": rule, "status": "unknown"} for rule in final.RULES],
              "findings": c.findings, "existing_qc": {"provenance_verified": False}}
    before = sum(f["occurrences"] for f in c.findings if f["status"] == "fail")
    final.prepare_machine_report(report)
    assert report["failure_points"] == [
        "Image data is not uint8: use type=matrix for floating-point or other numeric fields; if it is truly an image, use dtype=uint8 (left_tactile_data_pad, right_tactile_data_pad).",
        "Repository name must end with Demo_OXT.",
    ]
    assert sum(f["occurrences"] for f in report["findings"] if f["status"] == "fail") == before
    assert {f["code"] for f in report["findings"]} == {"image_not_uint8", "repo_suffix", "offline", "vlm_reported_warning"}
    warning = next(f for f in report["findings"] if f["code"] == "vlm_reported_warning")
    assert warning["examples"][0]["evidence"]["reason"] == ["基线记录不一致"]
    assert "provenance_verified" not in report["existing_qc"]
    markdown = final.report_markdown(report)
    assert markdown.splitlines()[:2] == ["1. " + report["failure_points"][0], "2. " + report["failure_points"][1]]
    assert "#### warning · vlm_reported_warning" in markdown
    assert "基线记录不一致" not in markdown and r"\u57fa" in markdown
    chinese = final.report_markdown(report, "cn")
    assert "基线记录不一致" in chinese and "需人工" not in chinese
    assert "type 改为 matrix" in chinese and "dtype 改为 uint8" in chinese
    page = final.report_html(report)
    assert page.index("<ol class='fail-list'>") < page.index("Machine check details")
    assert "<details open><summary>Machine check details</summary>" in page
    assert "data-status='manual'" not in page and "Check image contents" not in page
    assert "需人工" not in page


def test_no_fail_does_not_hide_unfinished_machine_checks():
    c = final.Collector()
    c.add("10.2", "unknown", "offline", "Network check unavailable")
    report = {"rules": c.summary(), "findings": c.findings}
    final.prepare_machine_report(report)
    markdown = final.report_markdown(report)
    assert markdown.startswith("No fail findings.\n\n---")
    assert "#### unknown · offline" in markdown
    assert "Remote checks were skipped in offline mode" in markdown
    assert report["overall_status"] == "needs_review"
    assert report["submission_ready"] is False


def test_english_outputs_and_chinese_copy_preserve_original_evidence(tmp_path):
    c = final.Collector()
    evidence = {"original_key": "原始名称", "original_value": "中文说明"}
    c.add("10", "fail", "metadata_chinese", "metadata 含中文", key="description", evidence=evidence)
    report = {"input_dir": str(tmp_path), "dataset": "Demo", "canonical_name": "Demo", "generated_at": "now",
              "rules": c.summary(), "findings": c.findings, "remote_repositories": [],
              "canonical_resolution": {"error": "无法唯一定位正式表格名称"}}
    final.save_report(report)
    for name in ("precheck_final_report.md", "precheck_final_report.html", "precheck_final_report.json"):
        assert not final.CJK_RE.search((tmp_path / name).read_text(encoding="utf-8"))
    chinese = (tmp_path / "precheck_final_report_cn.md").read_text(encoding="utf-8")
    assert chinese.startswith("1. metadata 含中文。") and "中文说明" in chinese
    stored = json.loads((tmp_path / "precheck_final_report.json").read_text(encoding="utf-8"))
    assert stored["findings"][0]["examples"][0]["evidence"] == evidence
    assert stored["overall_status"] == "fail" and stored["language"] == "en"
    assert not final.CJK_RE.search(stored["canonical_resolution"]["error"])
    assert "## Check coverage" in final.report_markdown(report)
    assert final.failure_points(report, "cn") == ["metadata 含中文。"]


def test_cli_help_is_english(capsys):
    with pytest.raises(SystemExit) as result:
        final.main(["--help"])
    assert result.value.code == 0
    output = capsys.readouterr().out
    assert not final.CJK_RE.search(output)
    assert "Chinese Markdown" in output and "--batch" in output
