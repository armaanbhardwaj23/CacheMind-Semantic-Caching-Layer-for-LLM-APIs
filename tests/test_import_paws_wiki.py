import importlib.util
from pathlib import Path


def _import_script():
    path = Path("scripts/import_paws_wiki.py")
    spec = importlib.util.spec_from_file_location("import_paws_wiki", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paws_importer_creates_balanced_provenance_marked_rows(tmp_path: Path) -> None:
    source = tmp_path / "paws.tsv"
    source.write_text(
        "id\tsentence1\tsentence2\tlabel\n"
        "a\tA one\tA two\t1\n"
        "b\tB one\tB two\t1\n"
        "c\tC one\tC two\t0\n"
        "d\tD one\tD two\t0\n"
    )

    rows = _import_script().build_rows(source, per_label=2, seed=42)

    assert len(rows) == 4
    assert sum(row["reuse_is_acceptable"] for row in rows) == 2
    assert {row["review_status"] for row in rows} == {
        "external_proxy_not_human_reviewed"
    }


def test_paws_importer_reads_jsonl_rows(tmp_path: Path) -> None:
    source = tmp_path / "paws.jsonl"
    source.write_text(
        "\n".join(
            '{"id": %d, "sentence1": "s%d", "sentence2": "t%d", "label": %d}' % (i, i, i, i % 2)
            for i in range(6)
        )
    )
    rows = _import_script().build_rows(source, per_label=3, seed=1)
    assert len(rows) == 6 and rows[0]["source_id"] == "1"
