# PITFALLS — 踩坑記錄與解法

> 每次遇到 bug / 設計錯誤，必須更新這份文件，避免重蹈覆轍。
> 格式：症狀 → 根本原因 → 修復方法 → 日期

---

## [P-001] Groq Key 輪換失效：永遠只用第一個 Key
- **症狀**：第一個 Groq Key 頻繁 429 Rate Limit，其他 Key 幾乎不被使用
- **根本原因**：`groq_client.py` 中 `_rotate_from_pool()` 直接 return `pool_keys[0]`，沒有實際輪換邏輯；`_pool_index` 字典初始化了但從未更新或讀取
- **修復**：改名為 `_rotate_pool_keys(pool_keys, task)`，使用 `_pool_index[task]` 追蹤輪換位置，每次調用後 index+1，並在 `generate()` 中使用回傳的 rotated list 取代原本的直接迭代
- **日期**：2026-03-01

---

## [P-002] researcher.py 呼叫私有方法 `_call_model_chain()`
- **症狀**：若 MultiModelClient 重構內部方法，researcher.py 會在運行時 AttributeError 崩潰
- **根本原因**：researcher.py 直接呼叫 `gh._call_model_chain(prompt)` 跨越封裝邊界
- **修復**：在 `MultiModelClient` 加入公開 `generate(prompt)` 方法；`_call_model_chain` 改為呼叫 `generate()` 的別名（向後相容）；researcher.py 統一改用 `gh.generate(prompt)`
- **日期**：2026-03-01

---

## [P-003] validator.py 誤判合法策略為 Look-Ahead（無 shift 警告）
- **症狀**：使用 `.diff()` / `.pct_change()` / `.ewm()` / `.rolling()` 的策略被誤標為「無時間偏移，可能有 look-ahead bias」，被 WARNING 誤導
- **根本原因**：`validate_code()` 只檢查 `.shift(` 是否存在，忽略其他等效的落後計算方式
- **修復**：改為檢查 `.shift(` 或 `.diff(` 或 `.pct_change(` 或 `.ewm(` 或 `.rolling(` 任一存在，只要有任何一種就視為有正確的時間偏移
- **日期**：2026-03-01

---

## [P-004] auto_runner.py `_consec_api_fail` 未在 `__init__` 初始化
- **症狀**：若第一次錯誤不是 API 全掛，直接跳到 `else` 分支執行 `self._consec_api_fail = 0` 是 OK 的，但若第一次就是 API 全掛，`getattr(self, '_consec_api_fail', 0)` 的 fallback 是隱藏問題
- **根本原因**：`_consec_api_fail` 只在錯誤路徑中隱式創建，不在 `__init__` 明確初始化，違反 Python 約定
- **修復**：在 `__init__` 中明確加入 `self._consec_api_fail = 0`
- **日期**：2026-03-01

---

## [P-005] Telegram 4000 字元硬截斷破壞報告格式
- **症狀**：報告被截斷在表格中間或分隔線內，接收者看到破碎的排版
- **根本原因**：`_send_telegram()` 使用 `report[:4000]` 硬截斷，不考慮行邊界
- **修復**：改為按行邊界分頁（PAGE_SIZE=3800）；多頁時加入 `[1/N]` 頁碼 header；頁之間 sleep(1) 避免 flood limit
- **日期**：2026-03-01

---

## [P-006] API 全掛偵測只匹配中文字串，英文錯誤被漏掉
- **症狀**：若 API error message 是英文（例如模型升級後訊息格式改變），冷卻邏輯不觸發，系統以正常速度繼續打 API 造成 burst
- **根本原因**：`if '都不可用' in result['error']` 只匹配硬編碼的中文字串
- **修復**：改為 `api_down_keywords = ['都不可用', 'all apis', 'api failed', 'no api keys']`，用 `.lower()` 統一後匹配任一關鍵字
- **日期**：2026-03-01

---

## [P-007] 無 git 自動化：策略進化結果無法追蹤血統
- **症狀**：每次策略演化後 `generated_strategies/` 和 `history_of_thoughts.json` 改變，但沒有 commit，歷史無法回溯
- **根本原因**：auto_runner.py 從未呼叫 git 命令
- **修復**：加入 `_git_commit(message, files)` 和 `_git_push()` 方法；在成功策略時 commit 策略文件；新最佳策略時觸發 push；每 N 輪週期報告後也 commit + push history
- **日期**：2026-03-01

---

## 待確認問題（尚未修復）

### [PENDING-001] backtest.py resample('ME') 版本相容性
- pandas 2.2+ 推薦 'ME'，但舊版不支援，可能出現 FutureWarning 或 ValueError
- **建議**：用 `try/except` 判斷 pandas 版本選擇 'ME' 或 'M'

### [PENDING-002] history_of_thoughts.json 無限增長
- 目前每次迭代追加，沒有清理或歸檔機制
- 建議：超過 10MB 時自動歸檔為 `history_archive_YYYYMM.json`

### [PENDING-003] Calmar Ratio 當 MaxDD=0 時回傳 0 而非 inf
- 對完美策略（理論上不可能）會產生誤導性的 Calmar=0
- 建議：改為 `float('inf')` 或用極小值替代

---

## 規則：未來加入新功能時的 Checklist

1. API 呼叫必須有 failover（Groq → GitHub Models → Gemini）
2. 所有屬性必須在 `__init__` 初始化，不依賴 `getattr` 隱式創建
3. 指標計算只能用落後計算（`.shift(n)` n≥1, `.rolling()`, `.diff()` 等）
4. 策略代碼通過 `StrategyValidator.validate_code()` 才能回測
5. 成功策略必須 git commit（包含策略文件 + history JSON）
6. 連續 3 次 API 失敗必須發送 Telegram 警報
7. 報告超過 4000 字元必須分頁發送
