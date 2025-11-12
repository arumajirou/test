# データ形式（unique_id, ds, y）

NeuralForecast は以下の縦長形式を推奨します：

| unique_id | ds        | y    |
|-----------|-----------|------|
| series_0  | 2024-01-01| 123.0|

## 自動変換ロジック（概要）
- 列名から `ds` 候補（`ds`, `date`, `datetime`, `timestamp`, `time`, `日付`, `日時`）を推定。
- 見つからない場合は **全列を datetime としてパースを試行**し、最初に成功した列を `ds` とする。
- 数値列が 2 列以上ある場合は **melt** して縦長化（列名が `unique_id` になる）。
- 数値列が 1 列のみの場合は `series_0` を `unique_id` として補う。
- 文字列数値は `to_numeric(..., errors="ignore")` で数値化を試みる。

## 注意点
- `ds` は `pandas.to_datetime(..., errors="coerce")` 後に `NaN` 行を除去。
- 出力は `unique_id, ds` でソートし、`y` は `float` にキャスト。
- タイムゾーン情報は利用ケースに応じて別途管理してください。
