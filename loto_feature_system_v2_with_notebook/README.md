# 宝くじ時系列特徴量生成システム（最小実装）

本パッケージは、仕様書の骨子に沿って **特徴量（歴史/将来/静的）** を自動生成する最小構成です。
- 入力: `examples/sample_loto.csv`（`unique_id, ds, y` の long 形式）
- 出力: `artifacts/features_hist.csv`, `artifacts/features_futr.csv`, `artifacts/features_stat.csv`

## クイックスタート

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 特徴量生成（入力CSV→出力CSV）
python scripts/run_pipeline.py --input examples/sample_loto.csv --output-dir artifacts --config config/pipeline_config.yaml
```

## 構成

```
loto_feature_system_v2/
├── config/
├── examples/               # サンプル入力（long形式）
├── artifacts/              # 生成物の保存先
├── scripts/                # CLIエントリポイント
├── src/
│   ├── core/               # ローディング/オーケストレータ
│   ├── pipelines/          # パイプライン定義
│   ├── utils/              # 付随ユーティリティ
│   └── integration/        # Ray/Dask等の統合面
└── tests/
```

## 仕様と前提
- 形式: long型 DataFrame（必須列: `unique_id`, `ds`, `y`）
- 頻度: 例では日次（`freq: D`）。`config/default_config.yaml` で変更可。
- 既知将来（カレンダー）・履歴ラグ/移動統計・静的属性の3群を生成します。
- 依存: `requirements.txt` を参照。
