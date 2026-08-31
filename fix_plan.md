# Task: refine パスに parakeet-ja 合議ゲートを実装する

## 背景 (測定済みの事実)

- `sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8` (models/ に取得済み) は
  クリーン音声で現行 ReazonSpeech Zipformer より正確 (FLEURS ja CER 6.6% vs 7.4%)
  だが、放送音声では字幕対象外の周辺発話・背景実況まで書き起こして崩壊する
  (実放送セット 31.2% vs 7.2%)。
- 両モデルの出力が近い (CER(hyp_z, hyp_p) <= 0.25) 発話だけ parakeet を採用する
  合議ゲートをシミュレートすると **FLEURS 5.9% / 実放送 5.8%** で両ドメイン同時改善
  (hayamimi-paper/experiments/results/ の candidate_*.json から再現可能)。

## 実装仕様

1. refine (清書) パスの ja 発話グループに対してのみ適用する。速報 (partial/final)
   経路は一切変更しない。
2. refine の再デコード時: 従来どおり Zipformer で清書 → 同じ音声を parakeet-ja
   (`OfflineRecognizer.from_nemo_ctc`) でもデコード → 両テキスト間の CER を
   `eval_accuracy.cer_ja` と同じ正規化で計算 → 閾値以下なら parakeet 出力を採用。
3. フラグ: `--refine-ja-second-opinion` (default off で入れ、検証後に default 判断) と
   `--refine-agree-threshold` (default 0.25)。
4. parakeet-ja モデルは refine で初めて必要になったときにロードし、既存の
   `--max-resident` LRU 会計に載せる (+625MB)。モデルが models/ に無い場合は
   警告を出してゲート無効 (従来動作) にフォールバック。
5. ITN・句読点・--replace は採用側テキストに従来どおり適用する。
6. ユニットテスト: ゲートの採用/棄却判定 (テキスト対を与えて閾値で分岐すること、
   モデル不在フォールバック) を tests/ に追加。

## 受け入れ条件 (これが全部通ったら完了)

worktree = このディレクトリ、検証ハーネス = ~/Desktop/Programing/hayamimi-paper

```bash
# 1) ユニットテスト全パス
~/Desktop/Programing/hayamimi/.venv/bin/python -m pytest tests/ -q

# 2) FLEURS 5言語 (refine ゲートを有効化した評価)
cd ~/Desktop/Programing/hayamimi-paper && HAYAMIMI_ROOT=<この worktree> \
  .venv/bin/python experiments/run_scorecard.py --out-suffix _pgate
# 判定: ja <= 0.062、en/ko/zh/yue は fleurs_scorecard_headfix2.json から +0.005 以内

# 3) 実放送 (realset manifest)
# 判定: ja <= 0.065
```

注: run_scorecard.py は RoutedASR.transcribe() を直接呼ぶため、ゲートを
transcribe 側にも通す評価用フックが必要なら「eval 用に refine 相当の
second-opinion を transcribe に適用するオプション」を RoutedASR に追加してよい
(本番の速報経路の default 動作を変えないこと)。

## 進捗ログ (イテレーションごとに追記)

- [ ] 実装
- [ ] ユニットテスト
- [ ] FLEURS 検証
- [ ] realset 検証
