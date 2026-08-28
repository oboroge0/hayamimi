#!/usr/bin/env python3
"""Start a local, recorded hayamimi meeting session on macOS.

The user chooses a meeting label and a currently connected physical output.
A temporary stacked Core Audio output mirrors meeting sound to that output and
BlackHole.  On exit the physical output is restored and the temporary device is
removed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent.parent
SWIFT_SOURCE = ROOT / "scripts" / "audio_profile.swift"
AUDIO_HELPER = ROOT / ".venv" / "bin" / "hayamimi-audio-profile"
SWIFT_MODULE_CACHE = ROOT / ".venv" / "cache" / "swift"
PROFILE_NAME = "hayamimi 会議 + BlackHole"
PROFILE_UID = "local.hayamimi.meeting-output"
DEFAULT_BLACKHOLE = "BlackHole 2ch"
DEFAULT_MIC = "MacBook Proのマイク"

APP_LABELS = {
    "teams": "Microsoft Teams",
    "zoom": "Zoom",
    "slack": "Slack",
    "discord": "Discord",
    "google-meet": "Google Meet",
    "other": "その他の会議",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="会議出力をBlackHoleへ複製し、hayamimiで録音・字幕保存します。"
    )
    parser.add_argument("--app", choices=APP_LABELS, help="会議サービス（省略時は選択）")
    parser.add_argument("--output", help="通常聞く物理出力のCore Audio名（省略時は選択）")
    parser.add_argument("--mic-device", default=DEFAULT_MIC, help="自分の物理マイク")
    parser.add_argument("--blackhole", default=DEFAULT_BLACKHOLE, help="BlackHole装置名")
    parser.add_argument(
        "--recordings-dir", type=Path, default=ROOT / "recordings",
        help="会議ごとのWAV・文字ログ保存先",
    )
    parser.add_argument("--port", type=int, default=8833, help="字幕ダッシュボードのポート")
    parser.add_argument("--no-open-dashboard", action="store_true", help="ブラウザを自動で開かない")
    parser.add_argument("--dry-run", action="store_true", help="切替や録音を行わず予定だけ表示")
    return parser


def command_output(command: list[str]) -> str:
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout.strip()


def switchaudio_path() -> str:
    path = shutil.which("SwitchAudioSource")
    if not path:
        raise RuntimeError(
            "SwitchAudioSourceがありません。`brew install switchaudio-osx`を実行してください。"
        )
    return path


def is_virtual_output(name: str, blackhole: str = DEFAULT_BLACKHOLE) -> bool:
    folded = name.casefold()
    return (
        name == blackhole
        or "blackhole" in folded
        or "hayamimi 会議" in name
        or name == "Meeting + BlackHole"
    )


def list_physical_outputs(switchaudio: str, blackhole: str) -> list[str]:
    output = command_output([switchaudio, "-a", "-t", "output"])
    seen: set[str] = set()
    physical = []
    for line in output.splitlines():
        name = line.strip()
        if name and name not in seen and not is_virtual_output(name, blackhole):
            physical.append(name)
            seen.add(name)
    return physical


def choose(title: str, values: list[tuple[str, str]]) -> str:
    if not values:
        raise RuntimeError(f"選択できる{title}がありません。先に装置を接続してください。")
    print(f"\n{title}を選んでください:")
    for number, (_, label) in enumerate(values, start=1):
        print(f"  {number}. {label}")
    while True:
        try:
            selected = int(input("> ").strip())
            return values[selected - 1][0]
        except (ValueError, IndexError):
            print(f"1〜{len(values)}で入力してください。")


def ensure_audio_helper() -> Path:
    if (
        AUDIO_HELPER.exists()
        and AUDIO_HELPER.stat().st_mtime >= SWIFT_SOURCE.stat().st_mtime
    ):
        return AUDIO_HELPER
    swiftc = shutil.which("swiftc")
    if not swiftc:
        raise RuntimeError(
            "Swiftコンパイラがありません。`xcode-select --install`を実行してください。"
        )
    AUDIO_HELPER.parent.mkdir(parents=True, exist_ok=True)
    SWIFT_MODULE_CACHE.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            swiftc,
            "-module-cache-path", str(SWIFT_MODULE_CACHE),
            str(SWIFT_SOURCE),
            "-framework", "CoreAudio",
            "-o", str(AUDIO_HELPER),
        ],
        check=True,
    )
    return AUDIO_HELPER


def session_directory(base: Path, app: str, now: dt.datetime | None = None) -> Path:
    now = now or dt.datetime.now()
    app_slug = app.replace("google-meet", "meet")
    candidate = base / now.strftime("%Y-%m-%d") / f"{now.strftime('%H%M%S')}_{app_slug}"
    suffix = 2
    while candidate.exists():
        candidate = candidate.with_name(f"{now.strftime('%H%M%S')}_{app_slug}_{suffix}")
        suffix += 1
    return candidate


def build_hayamimi_command(
    python: Path,
    blackhole: str,
    mic: str,
    transcript: Path,
    recording: Path,
    port: int,
) -> list[str]:
    return [
        str(python),
        str(ROOT / "scripts" / "realtime_transcribe.py"),
        "--translate", "ja",
        "--serve", str(port),
        "--device", blackhole,
        "--mic-device", mic,
        "--transcript", str(transcript),
        "--record", str(recording),
    ]


def set_output(switchaudio: str, name: str) -> None:
    subprocess.run([switchaudio, "-s", name, "-t", "output"], check=True)


def write_metadata(path: Path, app: str, physical_output: str, mic: str, blackhole: str) -> None:
    payload = {
        "started_at": dt.datetime.now().astimezone().isoformat(),
        "meeting_app": APP_LABELS[app],
        "physical_output": physical_output,
        "microphone": mic,
        "loopback": blackhole,
        "processing": "local",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    app = args.app or choose("会議サービス", list(APP_LABELS.items()))
    switchaudio = switchaudio_path()
    physical_outputs = list_physical_outputs(switchaudio, args.blackhole)
    if args.output:
        if args.output not in physical_outputs:
            available = "、".join(physical_outputs) or "なし"
            raise RuntimeError(
                f"出力 {args.output!r} は現在利用できません。接続中: {available}"
            )
        physical_output = args.output
    else:
        physical_output = choose("音を聞く装置", [(name, name) for name in physical_outputs])

    session_dir = session_directory(args.recordings_dir.expanduser(), app)
    transcript = session_dir / "meeting.txt"
    recording = session_dir / "meeting.wav"
    metadata = session_dir / "session.json"
    python = ROOT / ".venv" / "bin" / "python"
    command = build_hayamimi_command(
        python, args.blackhole, args.mic_device, transcript, recording, args.port
    )

    print("\n開始内容")
    print(f"  会議: {APP_LABELS[app]}")
    print(f"  聞く装置: {physical_output}")
    print(f"  保存先: {session_dir}")
    print("  処理: 完全ローカル（WAV、原文、日本語訳、清書）")
    if app == "google-meet":
        print("  Google Meetの既存字幕記録は併用可能です。hayamimiは音声録音と翻訳の独立記録になります。")
    if args.dry_run:
        print("  予行なので音声切替・録音は行いません。")
        return 0

    if not python.exists():
        raise RuntimeError(f"Python仮想環境がありません: {python}")

    helper = ensure_audio_helper()
    session_dir.mkdir(parents=True)
    transcript.touch(exist_ok=False)
    write_metadata(metadata, app, physical_output, args.mic_device, args.blackhole)
    child: subprocess.Popen | None = None
    profile_created = False

    def request_stop(_signum, _frame):
        raise KeyboardInterrupt

    for stop_signal in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(stop_signal, request_stop)
    try:
        subprocess.run(
            [str(helper), "create", physical_output, args.blackhole, PROFILE_NAME, PROFILE_UID],
            check=True,
        )
        profile_created = True
        set_output(switchaudio, PROFILE_NAME)
        print(f"\n会議用出力へ切り替えました: {PROFILE_NAME}")
        print("終了するにはこの画面で Control-C を押してください。終了後は通常出力へ戻ります。")
        child = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
        if not args.no_open_dashboard:
            time.sleep(1.0)
            subprocess.run(
                ["open", f"http://127.0.0.1:{args.port}/dashboard"],
                check=False,
            )
        returncode = child.wait()
        if returncode:
            print(f"hayamimiが終了コード{returncode}で停止しました。", file=sys.stderr)
        return returncode
    except KeyboardInterrupt:
        print("\n会議記録を終了しています…")
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.terminate()
                child.wait(timeout=10)
        return 0
    finally:
        try:
            set_output(switchaudio, physical_output)
            print(f"通常出力へ戻しました: {physical_output}")
        except subprocess.SubprocessError as exc:
            print(f"通常出力への復元に失敗しました: {exc}", file=sys.stderr)
        if profile_created:
            subprocess.run([str(helper), "destroy", PROFILE_UID], check=False)
        if session_dir.exists():
            print(f"保存先: {session_dir}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n開始前に中止しました。", file=sys.stderr)
        raise SystemExit(130)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        raise SystemExit(1)
