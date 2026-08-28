import datetime as dt
import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))

from meeting_session import (build_arg_parser, build_hayamimi_command,
                             is_virtual_output, session_directory)


def test_meeting_cli_accepts_all_supported_apps_and_defaults_to_interactive():
    parser = build_arg_parser()
    assert parser.parse_args([]).app is None
    for app in ("teams", "zoom", "slack", "discord", "google-meet", "other"):
        assert parser.parse_args(["--app", app]).app == app


def test_virtual_outputs_are_not_offered_as_listening_devices():
    assert is_virtual_output("BlackHole 2ch")
    assert is_virtual_output("Meeting + BlackHole")
    assert is_virtual_output("hayamimi 会議 + BlackHole")
    assert not is_virtual_output("MacBook Proのスピーカー")
    assert not is_virtual_output("外部ヘッドフォン")
    assert not is_virtual_output("My AirPods")


def test_session_directory_never_reuses_an_existing_meeting(tmp_path):
    now = dt.datetime(2026, 8, 28, 18, 30, 0)
    first = tmp_path / "2026-08-28" / "183000_teams"
    first.mkdir(parents=True)
    assert session_directory(tmp_path, "teams", now).name == "183000_teams_2"


def test_hayamimi_command_records_mixed_audio_and_japanese_translation(tmp_path):
    command = build_hayamimi_command(
        tmp_path / "python",
        "BlackHole 2ch",
        "MacBook Proのマイク",
        tmp_path / "meeting.txt",
        tmp_path / "meeting.wav",
        8833,
    )
    assert command[command.index("--translate") + 1] == "ja"
    assert command[command.index("--device") + 1] == "BlackHole 2ch"
    assert command[command.index("--mic-device") + 1] == "MacBook Proのマイク"
    assert command[command.index("--transcript") + 1].endswith("meeting.txt")
    assert command[command.index("--record") + 1].endswith("meeting.wav")
