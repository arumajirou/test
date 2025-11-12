# NF 外生特徴量ジェネレータ（DB駆動; futr_/hist_/stat_）

このZIPは、PostgreSQLのテーブルから長形式データ（`unique_id, ds, y, ...`）を取得して、
漏洩防止設計で **futr_ / hist_ / stat_** 接頭辞の特徴量を生成し、DBに書き戻す最小環境です。

## 構成
- `featgen/` … パッケージ本体
  - `providers_basic.py` … futr（カレンダー+フーリエ）、hist（ラグ/移動/差分）、Hampel異常検知
  - `db_utils.py` … DB接続（`db_config.py`を参照）
  - `config_default.yaml` … 既定設定（窓や周期）
- `cli_make_features_from_db.py` … 実行スクリプト（CLI）
- `config_default.yaml` … ルートにもコピー済み（編集しやすいように）
- `requirements.txt` … 必要パッケージ（最小）。任意の追加はコメント解除。
- `db_config.py.template` … `db_config.py` のひな形

## セットアップ
```bash
python -m pip install -r requirements.txt
cp db_config.py.template db_config.py
# db_config.py を編集（認証情報、TABLE_PREFIX など）
```

## 実行
```bash
python cli_make_features_from_db.py config_default.yaml
```
- 入力テーブル（既定）: `TABLE_PREFIX + "loto_final"`
- 出力テーブル（既定）: `TABLE_PREFIX + "features_hist"`, `TABLE_PREFIX + "features_futr"`
- 書込モードは `config_default.yaml` の `io.write_mode` で制御（`replace|append`）

## メモ
- すべての `hist_` は **左閉窓 + shift(1)** で計算し、将来情報の混入を防ぎます。
- 追加の特徴ライブラリ（tsfresh/TSFEL/ROCKET/stumpy 等）は `requirements.txt` のコメントを外して導入し、
  プロバイダを拡張する設計に対応しています（コードは `featgen/providers_basic.py` を参考に追加）。
