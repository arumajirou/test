# Ray on Windows の既知問題
Windows 環境では以下の症状が報告されています：

- ダッシュボード/エージェントのロギング初期化で例外（`ValueError: Must have exactly one of read or write mode`）
- `dashboard_agent failed and raylet fate-shares with it` により raylet が即時終了
- `Unable to register worker with raylet` が発生し Tune が実行不能

## 回避策
- **Linux（Kubuntu）での実行を推奨**。Ray/Tune が安定して動作します。
- どうしても Windows の場合は、`local_mode=True`, `include_dashboard=False` 等を設定しても改善しないケースがあるため、
  基本方針としては Linux への移行を推奨します。
