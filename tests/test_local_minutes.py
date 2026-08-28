import json
import os
import sys

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))

from local_minutes import (apply_glossary_rules, build_arg_parser,
                           clean_transcript, extract_json_object)


def test_clean_transcript_drops_saved_japanese_translations_and_numbers_lines():
    source = "Hello.\n  →ja こんにちは。\nNext line.\n"
    assert clean_transcript(source) == "[L0001] Hello.\n[L0002] Next line."


def test_extract_json_object_accepts_fenced_model_output():
    result = extract_json_object('```json\n{"目的": [{"内容": "確認"}]}\n```')
    assert result == {"目的": [{"内容": "確認"}]}


def test_extract_json_object_rejects_missing_or_non_object_json():
    with pytest.raises(ValueError, match="JSON"):
        extract_json_object("抽出できませんでした")
    with pytest.raises(ValueError, match="JSON"):
        extract_json_object(json.dumps(["配列"]))


def test_apply_glossary_rules_replaces_terms_and_removes_uncertain_lines():
    text = "製品はStraticsです。\n- 不明語 rushed in dual を確認する。\n次の行。"
    glossary = "置換: Stratics => Stratyx\n削除行: rushed in dual\n"
    assert apply_glossary_rules(text, glossary) == "製品はStratyxです。\n次の行。"


def test_local_minutes_cli_defaults_to_local_model_and_refuses_implicit_focus():
    args = build_arg_parser().parse_args(["meeting.txt"])
    assert args.model == "models/qwen3.8-27b-4bit"
    assert args.focus == ""
    assert args.glossary is None
    assert args.evidence is None
    assert args.force is False
