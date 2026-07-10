from src.vlm_checker import check_dataset


def test_missing_dir():
    report = check_dataset("non-existent-input-folder")
    assert report["exists"] is False
    assert report["status"] == "missing_input_dir"
