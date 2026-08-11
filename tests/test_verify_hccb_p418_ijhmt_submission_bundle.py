from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/verify_hccb_p418_ijhmt_submission_bundle.py"
REFRESH = ROOT / "code/run_hccb_p418_manuscript_refresh.sh"
sys.path.insert(0, str(ROOT / "code"))

from verify_hccb_p418_ijhmt_submission_bundle import resolve_record_path


def test_bundle_verifier_checks_standalone_source_and_pdf_text() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "archive.testzip()" in text
    assert "latexmk" in text
    assert "pdflatex+bibtex" in text
    assert 'require_tool("pdflatex")' in text
    assert 'require_tool("bibtex")' in text
    assert 'parser.add_argument("--texinputs"' in text
    assert 'parser.add_argument("--record"' in text
    assert "pdfinfo" in text
    assert "pdftotext" in text
    assert "source rebuild page mismatch" in text
    assert "source rebuild text differs materially" in text
    assert "text_match_ratio < 0.999" in text
    assert 'len(figure_members) != 7' in text
    assert '"supplement_included": False' in text


def test_bundle_verifier_accepts_preflight_package_record_name(tmp_path: Path) -> None:
    record = tmp_path / "package_record.json"
    record.write_text("{}\n", encoding="utf-8")
    assert resolve_record_path(tmp_path, None) == record


def test_bundle_verifier_prefers_formal_record_name(tmp_path: Path) -> None:
    formal = tmp_path / "record.json"
    preflight = tmp_path / "package_record.json"
    formal.write_text("{}\n", encoding="utf-8")
    preflight.write_text("{}\n", encoding="utf-8")
    assert resolve_record_path(tmp_path, None) == formal


def test_formal_refresh_requires_bundle_verification() -> None:
    text = REFRESH.read_text(encoding="utf-8")
    assert "verify_hccb_p418_ijhmt_submission_bundle.py" in text
    assert "--require-complete" in text
    assert "submission_bundle_verification.json" in text
    assert "runtime/p418_texmf" in text
