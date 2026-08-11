from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def manuscript_title() -> str:
    text = (ROOT / "manuscript/main.tex").read_text(encoding="utf-8")
    match = re.search(r"\\title\{([^}]+)\}", text)
    assert match is not None
    return match.group(1).replace("--", "–")


def test_repository_metadata_uses_current_manuscript_title_and_licences() -> None:
    expected = manuscript_title()
    metadata = json.loads(
        (ROOT / "reproducibility/repository_release_metadata_draft.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["title"] == expected
    assert metadata["software_license"] == "MIT"
    assert metadata["data_license"] == "cc-by-4.0"
    assert metadata["pending_author_fields"] == []
    assert metadata["repository_url"] == (
        "https://github.com/wangjianfttt/fusion-pebble-bed-heat-transfer-ai"
    )


def test_citation_file_uses_current_title_and_software_licence() -> None:
    expected = manuscript_title()
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert f'title: "{expected}"' in text
    assert "license: MIT" in text
    assert (
        'repository-code: "https://github.com/wangjianfttt/'
        'fusion-pebble-bed-heat-transfer-ai"'
    ) in text
