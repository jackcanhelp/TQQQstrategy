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

---

## [P-008] strategy_id 永遠是同一個值（例如 673）
- **症狀**：每次迭代都顯示 "Running iteration 673..."，total_iterations 不遞增
- **根本原因**：`run_single_iteration()` 的 `except Exception` 路徑只設定 `result['error']`，沒有呼叫 `_record_failure()`，導致 `total_iterations` 永遠不遞增，`get_next_strategy_id()` 一直回傳同一個值
- **修復**：在 try 前初始化 `idea = "N/A"`，在 except 末尾也加 `_record_failure(strategy_id, idea, result['error'])`
- **日期**：2026-03-01

---

## [P-009] LLM 生成錯誤的 import（`from BaseStrategy import ...`）
- **症狀**：`ModuleNotFoundError: No module named 'BaseStrategy'`（或 'BaseStra' 截斷版）
- **根本原因**：LLM 看到 `class Foo(BaseStrategy)` 就自作聰明寫 `from BaseStrategy import BaseStrategy`（把類名當模組名），正確應是 `from strategy_base import BaseStrategy`
- **修復**：加入 `_fix_imports()` 方法，用 regex 移除錯誤 import 並確保正確 import 在頂端；prompt 明確標注 `Import EXACTLY: from strategy_base import BaseStrategy`
- **日期**：2026-03-01

---

## [P-010] LLM 生成 `__init__(self, data)` 導致實例化失敗
- **症狀**：`TypeError: __init__() missing 1 required positional argument: 'data'`
- **根本原因**：`StrategySandbox.load_strategy()` 用 `strategy_class()` 無參數實例化，但 LLM 看到 `init(self, data)` 和 `BaseStrategy` 就誤寫成帶參數的 `__init__(self, data)`
- **修復**：在 prompt 的 EXAMPLE STRUCTURE 中明確展示 `def __init__(self): super().__init__()`，說明 `__init__` 不接受任何參數
- **日期**：2026-03-01

---

## [P-011] backtest.py 呼叫 `strategy.validate_signals()` 但策略未繼承 BaseStrategy
- **症狀**：`AttributeError: 'Strategy_GenN' object has no attribute 'validate_signals'`
- **根本原因**：`BacktestEngine.run()` 無條件呼叫 `strategy.validate_signals()`，但若 LLM 生成的類別沒有正確繼承 BaseStrategy（可能因 import 失敗），此方法不存在
- **修復**：`backtest.py` 改用 `hasattr` 檢查，無方法時 fallback 到 `raw_signals.clip(-1, 1).fillna(0)`
- **日期**：2026-03-01

---

---

## [P-012] SyntaxError 被外層 except 吞掉，永遠不進 fix 路徑
- **症狀**：`Syntax error in generated code: invalid syntax` 後直接記錄失敗，沒有嘗試修復
- **根本原因**：`sandbox.load_strategy()` 遇到 SyntaxError 會 raise Exception；在 `run_single_iteration()` 中，`load_strategy()` 直接在主 try 塊裡被呼叫，一旦 raise 就跳到最外層 except，完全繞過 `if not success: fix_strategy_code()` 路徑
- **修復**：用 inner try/except 包住 `load_strategy()` + `test_strategy()`，將任何 Exception 轉換為 `success=False, error=str(e)` 的形式，讓 fix 路徑正常觸發
- **日期**：2026-03-01

---

---

## [P-013] LLM 生成斷行 import 導致 SyntaxError
- **症狀**：`from strategy_base \nimport pandas as pd`（import 被斷成兩行），Python 語法錯誤
- **根本原因**：`_fix_imports()` 舊版只清除 `from BaseStrategy ...` 等明確錯誤，沒有清除 `from strategy_base` 的所有變體（含斷行形式）
- **修復**：`_fix_imports()` 改為「先核爆所有 from strategy_base 行，再重建正確 import」；`generate_strategy_code()` 加入 `ast.parse()` 預驗證，發現 SyntaxError 立即觸發 fix 而非等到 sandbox 才報錯
- **日期**：2026-03-01

---

## [P-022] Look-ahead bias 偵測不完整——多種未來資料存取方式未被攔截
- **症狀**：策略通過靜態驗證但實際使用未來資料（例如 `.pct_change(-1)`、`data['Close'].max()`、`rolling(center=True)`）
- **根本原因**：舊版 `LOOKAHEAD_PATTERNS` 只偵測 `.shift(-n)` 和幾個特定模式，漏掉多種常見 LLM 錯誤
- **修復**：全面擴充 `validator.py` 偵測規則（HARD / SOFT 分級）：
  - **HARD（直接拒絕）**：`shift(-N)`、`pct_change(-N)`、`diff(-N)`、`shift(periods=-N)`、`data['Close'].max()`、`.quantile()`（全域）、`.mean()`（全域）、`.std()`（全域）、`rolling(center=True)`、變數名 `tomorrow`/`next_bar`/`future_`/`look_ahead`
  - **SOFT（警告）**：`expanding().max/min()`、`nlargest/nsmallest`、`sort_values+head/tail`
  - 更新 `validate_code()` 為 3-tuple 格式（pattern, severity, message）
  - 拒絕時 auto_runner 印出具體違規訊息
  - code gen 和 fix prompt 均加入詳細的 ✅/❌ 對照表
- **日期**：2026-03-01

---

## [P-021] LLM import 未安裝的 TA 函式庫（talib, ta, pandas_ta）
- **症狀**：`Failed to load strategy: No module named 'talib'`（或 ta, pandas_ta）
- **根本原因**：LLM 在訓練資料中看過這些常見 TA 庫，但本環境未安裝
- **修復**：
  1. `_fix_imports()` 加入 FORBIDDEN_TA_LIBS 清單，自動移除這些 import 並替換 `lib.Func(...)` 為 `# REMOVED_LIB_CALL.`（觸發 fix 路徑重寫純 pandas 版本）
  2. `generate_strategy_code` prompt 明確標注 `❌ FORBIDDEN imports: talib, ta, pandas_ta`
- **日期**：2026-03-01

---

## [P-020] 好策略定義不合理：負 Sharpe / 躺平策略標記為 ✅
- **症狀**：`✅ Sharpe: 0.00 Calmar: 0.00` 或 `✅ Sharpe: -0.37` 出現，誤導為好策略
- **根本原因**：`run_single_iteration()` 的成功判斷只基於「回測跑完了」，沒有品質門檻
- **修復**：
  1. `validator.py` 加入 `validate_quality()` 方法，定義 TQQQ 好策略的最低標準：
     - Sharpe ≥ 0.5（風險調整收益必須正且達標）
     - CAGR ≥ 5%（至少跑贏現金）
     - MaxDD ≥ -70%（TQQQ 買持 2022 年 -87%，以此為參考）
     - 在市比例 ≥ 2%（不能完全躺平）
  2. `auto_runner.py` 區分顯示：✅ = 品質通過 / 📊 = 技術成功但品質不足 / ❌ = 失敗
  3. `researcher.py record_result()` 加入 `quality_pass` 欄位，ranking/context 用它過濾
  4. git commit 只在 `quality_pass=True` 時觸發（避免垃圾塞滿 history）
- **日期**：2026-03-01

---

## [P-019] Signal length doubled (8072 != 4036) — pd.Series 缺少 index
- **症狀**：`Signal length (8072) != data length (4036)`，恰好是 2 倍
- **根本原因**：LLM 在計算 ADX 等指標時，用 `pd.Series(np.where(...))` 把 numpy array 包成 Series，但沒有指定 index。numpy array 預設使用整數 index（0, 1, 2, ...），與 datetime index 的原始數據相加時，pandas 取兩者的 **union index** → 長度翻倍
- **修復**：
  1. `_fix_code_structure()` 加入 regex：把 `pd.Series(var)` 替換為 `pd.Series(var, index=self.data.index)`（僅限簡單變數名，跳過 literal `[...]` 和已有 `index=` 的）
  2. `generate_strategy_code` 和 `fix_strategy_code` prompt 加入明確警告：numpy array 轉 Series 必須加 `index=self.data.index`
- **日期**：2026-03-01

---

## 待確認問題（尚未修復）

## [P-023] LLM 指標 Scale 不匹配 → 條件永遠不發動（never enters market）
- **症狀**：`Strategy never enters the market (time_in_market=0.000%)`；Signal max=0.0000
- **根本原因**：LLM 將 price-level 指標（TEMA/SMA，值域 $10–$100+）與微小固定閾值比較（如 `> 0.2`, `> 0.8`）。由於 TEMA 遠大於 0.2，crossover 條件（`TEMA.rolling(3).mean().shift(1) <= 0.2`）永遠 False，進場訊號永遠不觸發。
- **修復**：
  1. `generate_strategy_code` prompt 加入 SCALE CHECK 表格：說明各指標的值域與正確比較方法
  2. `fix_strategy_code` TIM fix section 加入 SCALE MISMATCH 為首要診斷項目
  3. `test_strategy` TIM 偵測時自動掃描策略屬性，回報 price-level 指標名稱與值域
  4. `_fix_code_structure` 加入 P-023 區塊：替換非標準欄位引用（`sim_vix_pctile` → 計算版）
- **日期**：2026-03-01

## [P-024] LLM 引用不存在的欄位（sim_vix_pctile、vix 等）→ KeyError
- **症狀**：`KeyError: 'sim_vix_pctile'`，策略無法載入
- **根本原因**：LLM 在 prompt 範例中看到 "Simulated_VIX" 指標名稱，自以為 DataFrame 裡有 `sim_vix_pctile` 欄位
- **修復**：
  1. `_fix_code_structure` P-023 區塊：自動偵測並替換 `data['sim_vix_pctile']` 等非標準欄位為 rolling std 計算版
  2. `generate_strategy_code` prompt 加入 COLUMN RESTRICTION 警告：只允許 Open/High/Low/Close/Volume
  3. `fix_strategy_code` KeyError section 已有處理，現在 _fix_code_structure 提前修復
- **日期**：2026-03-01

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
