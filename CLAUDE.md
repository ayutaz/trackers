# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

**trackers** は、Roboflow が提供するマルチオブジェクト追跡（MOT）アルゴリズムのクリーンルーム実装ライブラリ。Apache 2.0 ライセンスで、任意の検出モデルと組み合わせて使用可能。Python 3.10〜3.13 対応。

クリーンルーム要件: すべてのアルゴリズムはゼロから再実装する必要がある。GPL/AGPL のソースコードを参照してはならない。学術論文と許容ライセンス（Apache/MIT/BSD）の実装のみ参考にすること。

## 開発コマンド

```bash
# 開発環境セットアップ
uv sync --group dev

# テスト実行
uv run pytest                              # ユニットテスト + doctest
uv run pytest test/eval/test_clear.py      # 単一ファイル
uv run pytest -k "test_clear"              # パターンマッチ
uv run pytest -m integration               # 統合テストのみ（データDL ~50MB）
uv run pytest -m ""                        # 全テスト（unit + integration）

# リント・フォーマット
ruff check trackers/ test/                 # コードチェック
ruff format trackers/ test/                # 自動フォーマット
mypy trackers/                             # 型チェック
pre-commit run --all-files                 # 全フック実行

# ドキュメント
uv sync --group docs
uv run mkdocs serve                        # ローカルプレビュー
```

## アーキテクチャ

### トラッカー基底クラス (`trackers/core/base.py`)

- `BaseTracker`: `update(detections: sv.Detections) -> sv.Detections` を定義する抽象クラス
- `BaseTrackerWithFeatures`: フレーム画像も受け取る拡張版（`update(detections, frame)`）

### トラッカー実装

- **SORTTracker** (`core/sort/`): カルマンフィルタ + IoU マッチングによる単段階追跡
- **ByteTrackTracker** (`core/bytetrack/`): 二段階アソシエーション（高信頼度→低信頼度の順でマッチング）

両トラッカーとも定速モデルのカルマンフィルタを使い、`scipy.optimize.linear_sum_assignment`（ハンガリアン法）でマッチングを行う。状態ベクトルは `[x1, y1, x2, y2, vx, vy, vx2, vy2]` の 8 次元。

### 評価メトリクス (`trackers/eval/`)

TrackEval との数値的一致を設計目標としたメトリクス群:
- **CLEAR MOT**: MOTA, MOTP, IDスイッチ等
- **HOTA**: 検出・アソシエーション・位置精度（複数 IoU 閾値で評価）
- **Identity**: IDF1（ID一貫性メトリクス）

`evaluate_mot_sequence()` / `evaluate_mot_sequences()` が主要 API。MOT ファイル形式の I/O は `eval/io.py` で処理。

### データ構造

`supervision` ライブラリの `sv.Detections` を中心に構築。検出結果は xyxy 形式の numpy 配列で、追跡 ID は `.tracker_id` フィールドに格納される（未マッチ/未成熟は −1）。

## コード規約

- **Docstring**: Google スタイル（必須）。doctest を含めること
- **型ヒント**: すべての関数に必須
- **Ruff**: ライン長 88、ルール E, F, I, A, Q, W, RUF, S を適用
- **McCabe 複雑度**: 最大 10
- PR は `develop` ブランチに対して作成

## テスト構成

- `test/eval/`: ユニットテスト（各メトリクスごと）
- `@pytest.mark.integration`: 実データでの統合テスト（SportsMOT, DanceTrack をダウンロードして TrackEval と数値比較）
- doctest は `--doctest-modules` で自動実行。外部データが必要な例には `# doctest: +SKIP` を付ける
- `test/conftest.py` にセッションスコープのフィクスチャでテストデータを `~/.cache/trackers-test/` にキャッシュ
