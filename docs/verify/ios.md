# iPhone 15 実機検証手順 (Mac)

`mobile/` (hayamimi_mobile) を iPhone 15 の実機に入れて動作確認するための手順。
これまでの検証はすべて Windows + Android エミュレータで行われており
(`docs/MOBILE.md`)、iOS 実機での検証は**まだ一度も行われていない** ——
このドキュメントはその最初の一歩を、迷わず再現できる形にするためのもの。

対象読者:

- このリポジトリのオーナー（開発経験はあるが iOS ビルドは初めて）
- Mac 側で作業する Claude Code セッション

Windows 側では Xcode の操作そのものを検証できないため、本ドキュメント中で
「Mac 未検証」と明記した箇所は、実際に Mac 上で一度通してみるまで手順の正確性
を保証しない。

## 検証ステータス: (a)〜(d) 完了、(e) 未検証 (2026-08-27)

この Mac 上の Claude Code セッションで、本ドキュメントの手順を実際に一度通した。
実機は iPhone 15 (おぼろげiPhone, iOS 27.0 beta)、Xcode 26.3、Flutter 3.47.1
(3.41.6 からアップグレード済み)。次にこの続きをやるセッション向けの引き継ぎ:

- **(a) Bench RTF**: 完了。ja int8, RTF **0.013**(PC int8 の 0.062 より
  約4.8倍速い — 想定通り ARM の int8 カーネルが効いた)。詳細・比較表は
  `docs/MOBILE.md` の「On-device (iPhone 15) verification」節。
- **(b) マニフェスト一括評価**: 完了。`testdata/eval_real/`(15 ja + 15 en、
  `scripts/make_realset.py --skip-eval` で Mac 上で再生成可能、公開HF
  データセット経由なので Windows 側の元データ不要)を実機に push して完走。
- **(c) Live 実マイク**: 完了。文字起こし成功、発熱なし、UI もたつきなし。
- **(d) 多言語ルーティング Live**: 完了(動作はした)。SenseVoice +
  whisper-tiny(LID)を実機に push、ja⇔en の言語バッジ切り替えを確認。
  ただし複数人が話している部屋での実施だったため、精度自体は参考値どまり
  — 静かな環境での単独話者での再検証が望ましい。`testdata/eval_real_zhko/`
  (zh/ko 用、`scripts/make_realset_zhko.py --skip-eval` で同様に再生成可能)
  も生成・push 済み。
- **(e) Remote**: **オーナー判断で対象外**(このセッションではやらない)。
  Mac 上で `python scripts/realtime_transcribe.py --input ws --serve
  --lang ja` は起動できた(`ws://<Mac IP>:8766/ingest`)が、この Mac の
  Wi-Fi (en0) の IP が通常のプライベート LAN 範囲外(`104.194.96.0/20`)で、
  iPhone から実際に届くか確認する前にオーナーが「Remote はやらなくていい」
  と判断し、検証対象から外れた。再開する場合はまず iPhone 側の Wi-Fi 設定
  (設定 > Wi-Fi > 接続中のネットワークの (i) > IP アドレス) を見て、Mac 側
  と同じセグメントかどうか確認するところから。

### このセッションで踏んだ落とし穴 (次回のために)

- **debug ビルドは `flutter run` / `flutter attach` 経由でしか起動できない**
  — `flutter install` で入れてから `xcrun devicectl device process launch`
  で単独起動しようとすると "Debug mode Flutter apps can only be launched
  from Flutter tooling" で弾かれる。
- **iOS 27 beta では LLDB のアタッチが遅い/不安定**(1〜3分、時にはそれ以上)。
  `flutter config --no-enable-lldb-debugging` は逆に Xcode 経由の起動に
  フォールバックしてさらに悪化した(Automation 権限待ちでハング)ので、
  デフォルト設定のまま気長に待つのが結局一番安定した。
- **devicectl のトンネル接続がしばしば invalidate される**
  ("CoreDeviceError", "Lost connection to device")。詰まったら
  **USB ケーブルを一度抜き差しする**とトンネルが張り直されて解消することが
  多かった。
- **`flutter run`/`flutter install` を繰り返すと、アプリの Documents
  コンテナ (push したモデルファイル一式) がリセットされることがある** —
  毎回起動後は `xcrun devicectl device info files --domain-type
  appDataContainer --domain-identifier dev.oboroge.hayamimiMobile
  --subdirectory Documents` で中身が残っているか確認し、消えていたら
  `xcrun devicectl device copy to` で再 push する。
- **ローカルネットワーク権限**(システム設定 > プライバシーとセキュリティ >
  ローカルネットワーク)をターミナルアプリに許可しないと、Dart VM Service
  の探索が `SocketException ... port 5353` で失敗し、アプリが起動時の
  一時停止状態のまま白画面になる。
- **モデルファイル配置は Finder より `xcrun devicectl device copy to
  --domain-type appDataContainer --domain-identifier
  dev.oboroge.hayamimiMobile` の方が速くて確実**(3節の Finder/Xcode
  Devices ウィンドウ手順は Mac 未検証のままだが、コマンド版で代替できる
  ことが分かった)。
- `mobile/ios/Runner/Info.plist` の `UIFileSharingEnabled` /
  `LSSupportsOpeningDocumentsInPlace` は既に設定済みだった(3.2節の
  「Mac 未検証」は解消 — 追加作業不要)。

### 未コミットの差分について

`mobile/ios/Runner.xcodeproj/project.pbxproj` に、Xcode で選択した個人の
Apple Developer Team ID (`DEVELOPMENT_TEAM = 9M9U5TB32V`) と
`objectVersion` の自動更新 (54→60) がローカルに残っている。他の開発者の
ビルドに影響しうる個人設定なので**意図的にコミットしていない**。このまま
残すか `git checkout -- mobile/ios/Runner.xcodeproj/project.pbxproj` で
戻すかは要判断。

## 0. 全体の流れ

1. Mac の前提環境を整える
2. リポジトリを取得してビルドする（無料 Apple ID 署名で実機へ）
3. モデルファイルを実機の Documents に配置する
4. 検証メニュー (a)〜(e) を優先順に実施する
5. 結果を PC (Windows) 側の `docs/MOBILE.md` に持ち帰る
6. (任意) Mac 側で Claude Code を使う場合の注意

---

## 1. 前提環境 (Mac)

- **Xcode** — App Store から最新版をインストール（Command Line Tools も同梱）。
  無料 Apple ID 署名を使う場合、Xcode 側のバージョンは特にこだわらず最新でよい。
- **Flutter SDK** — `mobile/pubspec.yaml` の `environment.sdk: ^3.13.1`
  (Dart SDK 制約) を満たすもの。このリポジトリでの Windows/Android 検証は
  Flutter **3.47.1** で行われている（`docs/MOBILE.md`）ので、同じマイナー系列
  を Mac 側でも使うと差分要因を減らせる。`flutter doctor` で iOS 関連の項目
  (Xcode, CocoaPods) が緑になっていることを確認する。
- **CocoaPods** — `sudo gem install cocoapods` (または Homebrew:
  `brew install cocoapods`)。`flutter doctor` の "CocoaPods" 項目が✗の場合は
  ここでインストールする。
- **iPhone 15 実機** — Lightning/USB-C ケーブルで Mac に接続し、初回は
  「このコンピュータを信頼しますか？」に同意する。

### 1.1 無料 Apple ID 署名で実機に入れる (Mac 未検証)

有料の Apple Developer Program に入っていなくても、無料 Apple ID で
「7日間だけ動く」実機ビルドは可能。手順（一般的な Flutter/Xcode の作法。
このリポジトリの `Runner.xcodeproj` に対する具体的な操作は Mac 未検証）:

1. `cd mobile && open ios/Runner.xcworkspace` で Xcode を開く（`.xcodeproj`
   ではなく **`.xcworkspace`** を開くこと — CocoaPods 経由で `sherpa_onnx_ios`
   の XCFramework を含んでいるため、`.xcodeproj` 単体を開くとリンクエラーに
   なる）。
2. 左のナビゲータで `Runner` ターゲット → **Signing & Capabilities** タブ。
3. **Team** に自分の無料 Apple ID を選択（未追加なら
   Xcode > Settings > Accounts で Apple ID を追加してから選び直す）。
4. **Bundle Identifier** を変更する。デフォルトは
   `mobile/ios/Runner.xcodeproj/project.pbxproj` に定義されている
   `PRODUCT_BUNDLE_IDENTIFIER` (`dev.oboroge.hayamimiMobile` 系、`.RunnerTests`
   variant あり)。無料 Apple ID は他人と重複しない Bundle ID しか使えないため、
   末尾に適当な文字列を足す（例: `dev.oboroge.hayamimiMobile.verify`）。
   `project.pbxproj` を直接 grep すれば現在値を確認できる:
   `grep PRODUCT_BUNDLE_IDENTIFIER ios/Runner.xcodeproj/project.pbxproj`
5. 実機を選択した状態で **▶ (Run)**、または後述の `flutter run -d <device-id>`
   でビルド・インストールする。
6. 初回起動時、iPhone 側で「信頼されていないデベロッパ」の警告が出る場合、
   **設定 > 一般 > VPNとデバイス管理** から該当の Apple ID プロファイルを
   信頼する。
7. **デベロッパモード** — iOS 16 以降は初回のデバイス上ビルド実行後、
   端末を再起動して **設定 > プライバシーとセキュリティ > デベロッパモード**
   を ON にする必要がある（ON にしないと署名済みアプリの起動がブロックされる）。
8. 無料署名のプロビジョニングプロファイルは**7日で失効する**。7日を超えて
   使い続ける場合は同じ手順（4〜6）を再実行してビルドし直す。

---

## 2. リポジトリ取得とビルド

```bash
git clone https://github.com/oboroge0/hayamimi.git
cd hayamimi/mobile
flutter pub get
```

`flutter pub get` は `hayamimi_core`（同リポジトリ内 `mobile/hayamimi_core/`
への path 依存、`mobile/pubspec.yaml` 参照）も解決する。

### 2.1 iOS ビルド

```bash
flutter build ios --debug --no-codesign   # まずビルドだけ通るか確認
# 実機に入れるには Xcode の ▶ か:
flutter run -d <device-id>                # `flutter devices` で iPhone の ID を確認
```

`sherpa_onnx` は iOS 向けに `sherpa_onnx_ios` パッケージ経由でプリビルドの
XCFramework を配布しており（`mobile/README.md` に既述）、`pod install` は
`flutter build`/`flutter run` の中で自動的に走る。**ここが最初につまずき
やすい箇所**（すべて Mac 未検証、事前に把握しておくと良い既知の注意点）:

- **初回 `pod install` は数分かかる** — XCFramework のダウンロード/展開が
  入るため、初回ビルドだけ通常より長くなる。ネットワークが不安定だと
  途中で失敗することがあるので、失敗したら `cd ios && pod install --repo-update`
  を単体で再実行してから `flutter run` をやり直す。
- **CocoaPods のバージョン不整合** — `flutter doctor` で CocoaPods が古いと
  警告が出ることがある。`pod --version` を確認し、必要なら
  `sudo gem install cocoapods` で更新する。
- **`Podfile.lock` の競合** — `mobile/ios/Podfile.lock` はリポジトリに
  コミットされていない想定（Android と同様、iOS 側もビルド成果物は
  `.gitignore` 対象）。初回は素直に `pod install` に生成させる。
- **署名エラー ("No profiles for ... were found")** — 上の 1.1 節の
  Bundle ID / Team 設定が反映されていない状態でビルドすると出る。
  Signing & Capabilities タブで赤いエラーが出ていないか確認する。

### 2.2 未検証事項（正直な現状）

`mobile/README.md` の "Building on macOS (iOS)" 節に書かれている手順
（`open ios/Runner.xcworkspace` → 初回だけ Xcode で署名設定 →
`flutter build ios --debug --no-codesign` / `flutter run`）は**まだ実際に
Mac 上で実行されたことがない**。sherpa_onnx の XCFramework 取得や
Podfile の解決が机上の想定どおりに進むかは、このドキュメントに沿って
最初に通した人（オーナー本人、または Mac 側の Claude Code セッション）が
確認して `docs/MOBILE.md` に追記するのが望ましい（下記「5. 結果の持ち帰り方」
参照）。

---

## 3. モデル配置

iOS には `adb` に相当するファイルプッシュ手段がない。代わりに **Finder /
Files アプリ経由**、または **Xcode の Devices and Simulators ウィンドウ**
から、アプリの Documents ディレクトリ
(`getApplicationDocumentsDirectory()`、iOS では
`<App>/Documents/`) に直接ファイルをコピーする。

### 3.1 必要ファイル一覧

`mobile/README.md`・`docs/MOBILE.md`・`scripts/asr_engine.py` の実体から
拾った、モデルディレクトリ名とファイル名（すべて `models/` 配下、
`scripts/download_models.py` が取得するのと同じ構成）:

| 用途 | ディレクトリ (PC 側 `models/` 配下) | 実機に置くファイル | 配置先 |
|---|---|---|---|
| ja ASR (ReazonSpeech, 必須) | `sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17/` | `encoder-*.int8.onnx`, `decoder-*.int8.onnx`, `joiner-*.int8.onnx`, `tokens.txt` の4点 | `Documents/model/` |
| VAD (Live 画面に必須) | `models/` 直下 | `silero_vad.onnx` | `Documents/vad/silero_vad.onnx` |
| en/zh/ko/yue ASR (言語ルーティング使用時のみ) | `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/` | `model.int8.onnx`, `tokens.txt` | `Documents/sense_voice/`（任意のパス。Live 画面のフィールドで指定） |
| LID プローブ (言語ルーティング使用時のみ) | `sherpa-onnx-whisper-tiny/` | `tiny-encoder.int8.onnx`, `tiny-decoder.int8.onnx` | `Documents/lid/`（任意のパス） |
| Bench 用テスト音声 | `testdata/ja_test.wav` | `ja_test.wav`（または任意の16kHz mono wav） | `Documents/test.wav` |

補足:

- ReazonSpeech の実ファイル名は sherpa-onnx のリリース物によって
  `encoder-epoch-99-avg-1.int8.onnx` のような命名になっており、
  厳密なファイル名一致は要求されない —
  `mobile/hayamimi_core/lib/bench/model_file_resolver.dart` が
  ディレクトリ内のファイルを `encoder`/`decoder`/`joiner`/`tokens.txt` の
  部分一致で拾い、`int8` を含むものを優先する。フォルダに fp32 と int8 が
  混在していても int8 が自動的に選ばれる。
- SenseVoice / whisper-tiny のパスは、Live 画面で「言語ルーティング」を
  `ja + SenseVoice` に切り替えたときだけ入力欄が出る
  (`mobile/hayamimi_core/lib/routing/routing_profile.dart`,
  `mobile/lib/live/live_page.dart`)。ja 単独の検証だけなら不要。
- モデル本体は `models/` が git 管理外のため、PC (Windows) 側で事前に
  `python scripts/download_models.py --minimal` 相当を実行して手元に
  揃えておく必要がある（`--minimal` は ja/en コア一式、~1.1GB。
  SenseVoice/whisper-tiny も検証するなら `--minimal` 無しのフル実行、
  または個別ファイルの追加取得が要る）。

### 3.2 実機への配置手順 (Mac 未検証)

方法はどちらでも良い。両方とも「Documents フォルダへの共有をアプリ側で
許可している」ことが前提（`UIFileSharingEnabled` / `LSSupportsOpeningDocumentsInPlace`
を `Info.plist` に追加する必要がある可能性が高い —
`mobile/ios/Runner/Info.plist` を確認し、無ければ追加するのが最初の一手。
現状 `Info.plist` には `NSMicrophoneUsageDescription` のみ確認できており、
ファイル共有キーの有無は Mac 未検証）:

**方法A: Finder 経由**

1. iPhone を Mac に接続し、Finder のサイドバーから端末を選択。
2. **ファイル** タブでインストール済みの hayamimi_mobile アプリを選び、
   モデルファイル/フォルダをドラッグ＆ドロップする。
3. Finder 経由のドロップは Documents 直下への配置になるため、
   `model/`・`vad/` のようなサブフォルダ構成にしたい場合は、事前に
   Mac 側でフォルダ構造を作ってからまとめてドロップするか、アプリ内の
   パス入力欄をアプリが実際に作った配置に合わせて調整する。

**方法B: Xcode の Devices and Simulators ウィンドウ**

1. Xcode メニュー > **Window > Devices and Simulators**。
2. 左で iPhone を選択 → **Installed Apps** で hayamimi_mobile を選ぶ →
   歯車アイコン (⚙️) から **Download Container...** で現在の内容を確認、
   または直接ファイルをドラッグして追加できる（Xcode バージョンにより
   UI が異なる）。

いずれの方法でも、配置後はアプリの Bench/Live 画面のパス入力欄（デフォルトは
`<Documents>/model/`, `<Documents>/vad/silero_vad.onnx`,
`<Documents>/test.wav` — `mobile/README.md` 記載のデフォルトと同一）を
実際に置いた場所に合わせて確認・編集する。

---

## 4. 検証メニュー（優先順）

### (a) Bench: ja int8 の RTF 計測

1. **Bench** タブを開く。
2. Model directory を `Documents/model/` に、WAV path を
   `Documents/test.wav` に設定（デフォルトのままなら変更不要）。
3. 実行して RTF (real-time factor = 処理時間 / 音声長) を記録する。

`docs/MOBILE.md` の PC/エミュレータ数値との比較用テンプレ:

| 環境 | RTF (ja int8, `modified_beam_search`) | 備考 |
|---|---|---|
| PC (Windows, x86-64) | 0.062 | `docs/MOBILE.md` INT8 量子化セクション |
| Android エミュレータ (x86_64, ホストPC上) | (informational only) | ホストCPU実行のため実機性能を反映しない |
| **iPhone 15 (実機, ここに記入)** | | |

`docs/MOBILE.md` が繰り返し強調している通り、エミュレータの数値は
「ホスト PC の CPU 上」の数値であり実機性能の代わりにならない。iPhone 15 の
数値が初めての「本物の ARM 実機速度」になる。

### (b) マニフェスト一括評価で CER パリティ再確認

Bench タブの「Manifest batch eval (debug only)」パネル（デバッグビルドのみ表示）
を使い、`testdata/eval_real/manifest.json` と対応する wav 一式を実機の
Documents に配置した上で一括デコードし、結果 JSON
(`wav`, `lang`, `ref`, `hyp`, `audio_s`, `decode_s`, `rtf`) を書き出す
(`mobile/hayamimi_core/lib/bench/manifest_eval_runner.dart`)。

- PC 側の `docs/MOBILE.md` 記載の CER: fp32/int8 とも 5.50%（full_int8）。
  Android エミュレータでの同一検証: 6.19%（+0.69pp）。
- iPhone 実機の結果 JSON を PC に持ち帰り
  (`scripts/eval_accuracy.py` の `cer_ja`) でスコアリングし、この2つの
  数値と比較する。1〜2pp 程度のズレは onnxruntime のカーネル差として
  想定内（`docs/MOBILE.md` の「On-emulator accuracy parity」節が同じ理屈で
  エミュレータの差を説明している）。

言語ルーティングまで検証するなら「Routed multilingual manifest eval
(debug only)」パネルで `testdata/eval_real/` + `testdata/eval_real_zhko/`
のサブセットを使う（`docs/MOBILE.md` の "Multi-language routing on mobile"
節、20クリップでの参考値: ja 4/5, en/zh/ko 5/5 の言語判定正解）。

### (c) Live 実マイク（清書・自動清書）

1. **Live** タブを開き、モデルパスを確認して **Start listening**。
2. 初回はマイク権限のダイアログが出る（`Info.plist` の
   `NSMicrophoneUsageDescription` — 既に設定済み）ので許可する。
3. 数文発話して VAD セグメントごとに文字起こしされることを確認する。
4. **清書** ボタンを押し、複数セグメントをまとめた再デコードが走ることを
   確認する。**自動清書** トグル（デフォルト OFF）も試す。
5. **連続10分程度**話し続け、以下を観察する（数値計測ではなく体感の
   観察ポイント）:
   - 端末の発熱（背面を触って「温かい」「熱い」の主観評価）。
   - バッテリー残量の減り方（開始時と終了時の % を記録）。
   - UI のもたつき（`mobile/README.md` に既述の通り、デコードは
     同期 FFI を `async` でラップしているだけなので、長いセグメントで
     一瞬 UI がカクつく可能性がある — 実機での体感を確認する）。

### (d) 多言語ルーティング Live

Live 画面の「Language routing」ドロップダウンを `ja + SenseVoice
(en/zh/ko/yue)` に切り替え、SenseVoice / LID モデルディレクトリのパスを
入力してから同様に発話する。日本語→英語のように話す言語を切り替え、
各行に付く言語バッジ（例: "JA", "EN"）が意図通りに切り替わるか確認する。
`docs/MOBILE.md` の "Open items" に書かれている通り、これまでルーティング
はマニフェスト一括評価でしか検証されておらず、**実際のマイクでの言語切り替え
はまだ一度も検証されていない** — この iOS 実機セッションが最初のライブ検証
になる可能性がある。

### (e) Remote（Wi-Fi 越しに PC の hayamimi へ）

1. PC (Windows) 側で `python scripts/realtime_transcribe.py --input ws --serve`
   を起動する。
2. iPhone とその PC が**同じ Wi-Fi ネットワーク**にいることを確認する。
3. **Remote** タブでサーバー URL を `ws://<PCのLAN IP>:8766/ingest` に設定し
   **接続**する（`mobile/README.md` "Remote mode" 節参照。エミュレータ用の
   `10.0.2.2` はここでは使わない — 実機なので PC の実 LAN IP が要る）。
4. 発話し、PC 側のフルパイプライン（5層言語ルーティング・清書・翻訳・
   話者ラベル）が反映された subtitle イベントが Remote 画面に届くことを
   確認する。

---

## 5. 結果の持ち帰り方

- (a)/(b) の計測 JSON・数値は AirDrop（または他の任意の手段）で PC に転送する。
- 追記先は `docs/MOBILE.md`:
  - (a) の RTF は「On-emulator accuracy parity」節の後、もしくは新しい
    "On-device (iPhone 15) verification" 節を追加してそこに記載するのが
    自然（このドキュメントの表テンプレをそのまま埋める形で良い）。
  - (b) の CER は同節の「Verification A」相当の比較表に実機の行を追加する。
  - (c)/(d) の発熱・電池・UI もたつきの所見、および (d) の「初めてのライブ
    言語切り替え検証」の結果は、"Open items" 節の該当項目を実測値で
    上書き・解消する形で追記する。
- Mac 側でこのドキュメント自体の記述ミス（コマンド・パス・ファイル名が
  実体と違う等）に気付いたら、このファイル (`docs/IOS_VERIFY.md`) 自体も
  合わせて修正する。

---

## 6. Mac 側で Claude Code を使う場合の注意

このリポジトリの検証作業を Mac 上の Claude Code セッションに任せる場合、
このリポジトリ（および運用者）の定石は以下の通り:

- **push / PR 作成は必ず人間の確認を経てから** — worktree 内でのコミット
  までは自律的に進めてよいが、`git push` や `gh pr create` はユーザーの
  明示的な承認なしに行わない。
- **ブランチ命名は `agent/<type>/<slug>`** — 例:
  `agent/feature/ios-verify`, `agent/test/ios-realmic`（`type` は feature /
  test / refactor / bugfix / docs）。このリポジトリの既存ブランチ
  (`agent/feature/mobile-app` 等) も同じ規則に従っている。
- **検証ゲート** — モデル/ルーティング/デコードパラメータに触れる変更は、
  `CONTRIBUTING.md` の方針どおり実音声での再評価（この場合は本ドキュメントの
  (a)〜(d)）を経てから統合する。テストだけでは不十分（`docs/BENCHMARKS.md`
  にある過去の regression 事例を参照）。
- **worktree は独立させる** — 複数の作業を並行させる場合、同一ブランチ・
  同一ファイルを別々のエージェントに同時に触らせない。
- 破壊的操作（`git push --force`、履歴改変、実機の初期化など）は行わない。
