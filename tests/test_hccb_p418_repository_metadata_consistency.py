from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


EXPECTED_CREATORS = [
    {
        "name": "Wang, Jian",
        "affiliation": (
            "Anhui University of Science and Technology; Institute of Plasma "
            "Physics, Hefei Institutes of Physical Science, Chinese Academy of Sciences"
        ),
    },
    {
        "name": "Wen, Wei",
        "affiliation": (
            "Anhui University of Science and Technology; Institute of Plasma "
            "Physics, Hefei Institutes of Physical Science, Chinese Academy of Sciences"
        ),
    },
    {
        "name": "Shen, Gang",
        "affiliation": "Anhui University of Science and Technology",
    },
    {
        "name": "Lei, Mingzhun",
        "affiliation": (
            "Institute of Plasma Physics, Hefei Institutes of Physical Science, "
            "Chinese Academy of Sciences"
        ),
    },
    {
        "name": "Song, Yuntao",
        "affiliation": (
            "Institute of Plasma Physics, Hefei Institutes of Physical Science, "
            "Chinese Academy of Sciences"
        ),
    },
]


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
    assert metadata["creators"] == EXPECTED_CREATORS


def test_citation_file_uses_current_title_and_software_licence() -> None:
    expected = manuscript_title()
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert f'title: "{expected}"' in text
    assert "license: MIT" in text
    assert (
        'repository-code: "https://github.com/wangjianfttt/'
        'fusion-pebble-bed-heat-transfer-ai"'
    ) in text
    cff_creators = [
        {"name": f"{family}, {given}", "affiliation": affiliation}
        for family, given, affiliation in re.findall(
            r"  - family-names: ([^\n]+)\n"
            r"    given-names: ([^\n]+)\n"
            r'    affiliation: "([^"]+)"',
            text,
        )
    ]
    assert cff_creators == EXPECTED_CREATORS


def test_manuscript_and_zenodo_draft_use_the_same_authors_and_licences() -> None:
    manuscript = (ROOT / "manuscript/main.tex").read_text(encoding="utf-8")
    manuscript_authors = [
        re.sub(r"\\corref\{[^}]+\}", "", name)
        for name in re.findall(r"^\\author\[[^]]+\]\{(.+)\}$", manuscript, re.MULTILINE)
    ]
    assert manuscript_authors == [
        "Jian Wang",
        "Wei Wen",
        "Gang Shen",
        "Mingzhun Lei",
        "Yuntao Song",
    ]
    assert "\\ead{wjfttt@mail.ustc.edu.cn}" in manuscript

    zenodo = json.loads(
        (
            ROOT
            / "results/hccb_p418_public_data_release_preflight/zenodo_metadata_draft.json"
        ).read_text(encoding="utf-8")
    )
    metadata = zenodo["metadata"]
    expected_title = f"Data and code for: {manuscript_title()}"
    assert metadata["title"] == expected_title
    assert metadata["creators"] == EXPECTED_CREATORS
    assert metadata["upload_type"] == "dataset"
    assert metadata["access_right"] == "open"
    assert metadata["license"] == "cc-by-4.0"
    assert zenodo["license_choice"] == {
        "software_license": "MIT",
        "data_license": "cc-by-4.0",
    }
