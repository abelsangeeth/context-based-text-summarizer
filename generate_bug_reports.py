"""
generate_bug_reports.py
=======================
Generates terminal-style bug-report images for all 13 QA test cases.

  bug_reports/pass/  -- fixed / post-patch state  (green header, PASS badge)
  bug_reports/fail/  -- bug-found / initial state  (red header,  FAIL badge)

Run : python generate_bug_reports.py
"""

import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
matplotlib.rcParams.update({"figure.dpi": 150})

# ── Colour palette ────────────────────────────────────────────────────────────
C_GREEN_HDR   = "#1a7a1a"
C_RED_HDR     = "#7a1a1a"
C_GREEN_PASS  = "#00cc44"
C_RED_FAIL    = "#cc2200"
C_ORANGE_FAIL = "#ff6600"
C_BLACK       = "#0d0d0d"
C_DARK_GREEN  = "#003300"
C_DARK_RED    = "#220000"
C_MONO_GREEN  = "#00ff41"
C_MONO_YELLOW = "#f0c040"
C_MONO_WHITE  = "#e8e8e8"
C_MONO_GREY   = "#888888"
C_MONO_RED    = "#ff4444"
C_META_GOLD   = "#f0c040"
MONO          = "DejaVu Sans Mono"
SANS          = "DejaVu Sans"

OUT_DIR  = os.path.join(os.path.dirname(__file__), "bug_reports")
PASS_DIR = os.path.join(OUT_DIR, "pass")
FAIL_DIR = os.path.join(OUT_DIR, "fail")
for d in (OUT_DIR, PASS_DIR, FAIL_DIR):
    os.makedirs(d, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
#  DATA
# ═════════════════════════════════════════════════════════════════════════════

PASS_RECORDS = [
    {
        "tc_id": "TC_001", "version": "V1.2", "sprint": "Sprint 1",
        "feature": "Text Input (US1)", "bug_id": "BUG_001",
        "title": "Word Count Not Updating - FIXED",
        "status": "PASS",
        "result_line": "Word count updated correctly: 500 words detected",
        "terminal_lines": [
            ("prompt",  "$ python app.py --test word_count"),
            ("normal",  "Initialising summariser UI..."),
            ("normal",  "Loading input handler module..."),
            ("normal",  "Pasting 500-word paragraph into textarea..."),
            ("warn",    "WARNING: onInput event listener was not bound on render"),
            ("normal",  "Applying patch: bind onInput to textarea on mount"),
            ("normal",  "Re-testing word count update..."),
            ("bar",     "[====================] 100%  3/3 checks passed"),
            ("success", "SUCCESS: Word count label shows 500 immediately after paste"),
        ],
        "expected":   "Word count updates in real time after paste",
        "actual":     "Word count now updates correctly - Bug fixed",
        "root_cause": "onInput event listener not bound to textarea on initial render",
        "resolution": "Closed", "date": "17-04-2026",
    },
    {
        "tc_id": "TC_002", "version": "V1.2", "sprint": "Sprint 1",
        "feature": "Text Input (US1)", "bug_id": "--",
        "title": "Word Count Updates Correctly - NO BUG",
        "status": "PASS",
        "result_line": "Word count validated: 50/50 words displayed correctly",
        "terminal_lines": [
            ("prompt",  "$ python app.py --test word_count_verify"),
            ("normal",  "Loading homepage..."),
            ("normal",  "Pasting sample paragraph (50 words)..."),
            ("normal",  "Reading word-count label from DOM..."),
            ("normal",  "Expected: 50   Actual: 50"),
            ("bar",     "[====================] 100%  1/1 checks passed"),
            ("success", "SUCCESS: Word count label matches expected value"),
        ],
        "expected":   "Word count updates in real time showing correct word count",
        "actual":     "Word count updates correctly immediately after paste",
        "root_cause": "No defect found",
        "resolution": "Closed", "date": "17-04-2026",
    },
    {
        "tc_id": "TC_003", "version": "V1.2", "sprint": "Sprint 1",
        "feature": "Text Input (US1)", "bug_id": "BUG_002",
        "title": "Empty Input Warning Not Shown - FIXED",
        "status": "PASS",
        "result_line": "Validation triggered: warning message displayed correctly",
        "terminal_lines": [
            ("prompt",  "$ python app.py --test empty_input_validation"),
            ("normal",  "Launching summariser homepage..."),
            ("normal",  "Leaving input area empty..."),
            ("normal",  "Clicking Summarise button..."),
            ("warn",    "BUG was: No warning shown - backend received null payload"),
            ("normal",  "Applying fix: frontend guard added before API call"),
            ("normal",  "Re-testing empty input scenario..."),
            ("bar",     "[====================] 100%  2/2 checks passed"),
            ("success", "SUCCESS: Warning 'Please enter text to summarise' shown"),
        ],
        "expected":   "Warning message: 'Please enter text to summarise'",
        "actual":     "Warning now displays correctly - fix verified",
        "root_cause": "Frontend validation skipped; backend received null payload",
        "resolution": "Verified for fix", "date": "17-04-2026",
    },
    {
        "tc_id": "TC_004", "version": "V1.2", "sprint": "Sprint 1",
        "feature": "Text Input (API)", "bug_id": "BUG_003",
        "title": "Word-Count Label Resets on Small Values - FIXED",
        "status": "PASS",
        "result_line": "Edge-case guard patched: label shows correctly for <10 words",
        "terminal_lines": [
            ("prompt",  "$ python app.py --test word_count_edge_case"),
            ("normal",  "Loading input handler..."),
            ("normal",  "Entering short text: 'Hello world' (2 words)..."),
            ("warn",    "BUG was: Word count label reset to 0 on small integer values"),
            ("normal",  "Root cause: int comparison guard reset label incorrectly"),
            ("normal",  "Applying patch: removed erroneous reset condition"),
            ("bar",     "[====================] 100%  3/3 edge cases passed"),
            ("success", "SUCCESS: Word count label shows 2 for 2-word input"),
        ],
        "expected":   "Word count label shows correctly for any input size",
        "actual":     "Label now updates correctly for small values - fix verified",
        "root_cause": "Edge-case guard using small integer values resets label to zero",
        "resolution": "Verified for fix", "date": "17-04-2026",
    },
    {
        "tc_id": "TC_005", "version": "V1.2", "sprint": "Sprint 1",
        "feature": "Text Input (US1)", "bug_id": "--",
        "title": "Clear Button Empties Input and Resets Count - PASS",
        "status": "PASS",
        "result_line": "Clear button: input emptied, word count reset to 0",
        "terminal_lines": [
            ("prompt",  "$ python app.py --test clear_button"),
            ("normal",  "Entering sample text in input area..."),
            ("normal",  "Word count before clear: 100"),
            ("normal",  "Clicking Clear button..."),
            ("normal",  "Reading input area value: '' (empty)"),
            ("normal",  "Reading word count label: 0"),
            ("bar",     "[====================] 100%  2/2 checks passed"),
            ("success", "SUCCESS: Input cleared and word count reset to 0 immediately"),
        ],
        "expected":   "Input area emptied; word count resets to 0",
        "actual":     "Input cleared and word count reset immediately",
        "root_cause": "No defect found",
        "resolution": "Closed", "date": "17-04-2026",
    },
    {
        "tc_id": "TC_006", "version": "V1.2", "sprint": "Sprint 2",
        "feature": "Copy to Clipboard (US6)", "bug_id": "BUG_004",
        "title": "Copy Button Still Clickable When Disabled - FIXED",
        "status": "PASS",
        "result_line": "Copy button correctly disabled and non-interactive before summary",
        "terminal_lines": [
            ("prompt",  "$ python app.py --test copy_button_state"),
            ("normal",  "Loading homepage with no summary generated..."),
            ("normal",  "Inspecting Copy button state..."),
            ("warn",    "BUG was: button.disabled = false despite greyed appearance"),
            ("normal",  "Applying fix: set disabled=true and pointer-events:none"),
            ("normal",  "Re-testing Copy button interactivity..."),
            ("bar",     "[====================] 100%  2/2 checks passed"),
            ("success", "SUCCESS: Copy button is disabled and non-interactive"),
        ],
        "expected":   "Copy button greyed / disabled before summary generated",
        "actual":     "Button now correctly disabled - fix verified",
        "root_cause": "Copy button visually disabled but remained interactive",
        "resolution": "Verified for fix", "date": "17-04-2026",
    },
    {
        "tc_id": "TC_007", "version": "V1.2", "sprint": "Sprint 2",
        "feature": "Copy to Clipboard (US6)", "bug_id": "BUG_005",
        "title": "Clipboard API Permission Failure - FIXED",
        "status": "PASS",
        "result_line": "Clipboard write succeeded with execCommand fallback",
        "terminal_lines": [
            ("prompt",  "$ python app.py --test clipboard_copy"),
            ("normal",  "Generating summary text..."),
            ("normal",  "Clicking Copy button..."),
            ("warn",    "ERROR was: navigator.clipboard.writeText() blocked (non-HTTPS)"),
            ("normal",  "Falling back to document.execCommand('copy')..."),
            ("normal",  "Verifying clipboard contents match summary text..."),
            ("bar",     "[====================] 100%  1/1 paste verification passed"),
            ("success", "SUCCESS: Summary text copied to clipboard via fallback"),
        ],
        "expected":   "Summary text copied to clipboard; paste reproduces the text",
        "actual":     "Clipboard write succeeded after applying execCommand fallback",
        "root_cause": "Clipboard API write blocked in non-HTTPS context",
        "resolution": "Added try/catch with execCommand fallback", "date": "19-04-2026",
    },
    {
        "tc_id": "TC_008", "version": "V1.2", "sprint": "Sprint 2",
        "feature": "Copy to Clipboard (US6)", "bug_id": "BUG_006",
        "title": "Toast Notification Not Shown After Copy - FIXED",
        "status": "PASS",
        "result_line": "Toast notification displayed for 3 seconds then dismissed",
        "terminal_lines": [
            ("prompt",  "$ python app.py --test toast_notification"),
            ("normal",  "Generating summary and clicking Copy..."),
            ("warn",    "BUG was: Toast component mounted; visibility flag not set"),
            ("normal",  "Applying fix: set toast.visible=true in copy callback"),
            ("normal",  "Adding 3-second auto-dismiss timer..."),
            ("normal",  "Toast visible at t=0s: True"),
            ("normal",  "Toast visible at t=3s: False (auto-dismissed)"),
            ("bar",     "[====================] 100%  1/1 toast checks passed"),
            ("success", "SUCCESS: Toast appears and auto-dismisses after 3 seconds"),
        ],
        "expected":   "Toast notification appears for 3 seconds then disappears",
        "actual":     "Toast now shows and auto-dismisses correctly - fix verified",
        "root_cause": "Toast visibility flag not set in copy success callback",
        "resolution": "Updated copy callback with 3s auto-dismiss timer", "date": "19-04-2026",
    },
    {
        "tc_id": "TC_009", "version": "V1.2", "sprint": "Sprint 3",
        "feature": "Usage History (US5)", "bug_id": "BUG_007",
        "title": "History Not Loading - FIXED",
        "status": "PASS",
        "result_line": "History loaded: 5/5 summaries with timestamps displayed",
        "terminal_lines": [
            ("prompt",  "$ python app.py --test history_load"),
            ("normal",  "Logging in with test account (5 prior summaries)..."),
            ("warn",    "ERROR was: 401 Unauthorized - stale auth token used"),
            ("normal",  "Applying fix: move History API call after token refresh"),
            ("normal",  "Token refreshed successfully"),
            ("normal",  "GET /api/history  ->  200 OK"),
            ("bar",     "[====================] 100%  5/5 history items loaded"),
            ("success", "SUCCESS: 5 summaries listed with correct timestamps"),
        ],
        "expected":   "History list shows 5 summaries with timestamps",
        "actual":     "History loads correctly after token-refresh fix",
        "root_cause": "History API called before auth token refresh; stale token used",
        "resolution": "Moved API call to after auth with updated headers", "date": "19-04-2026",
    },
    {
        "tc_id": "TC_010", "version": "V1.2", "sprint": "Sprint 3",
        "feature": "Usage History (US5)", "bug_id": "BUG_008",
        "title": "History Item Does Not Repopulate Input - FIXED",
        "status": "PASS",
        "result_line": "Input and output boxes repopulated from history entry",
        "terminal_lines": [
            ("prompt",  "$ python app.py --test history_repopulate"),
            ("normal",  "Opening history list..."),
            ("normal",  "Clicking on saved history entry..."),
            ("warn",    "BUG was: Output populated; input box remained empty"),
            ("normal",  "Applying DB migration: add input_text column to history table"),
            ("normal",  "Updating save handler to persist input + output"),
            ("normal",  "Re-testing history item click..."),
            ("bar",     "[====================] 100%  2/2 fields repopulated"),
            ("success", "SUCCESS: Both input and output restored from history"),
        ],
        "expected":   "Input and output boxes repopulated with saved data",
        "actual":     "Both fields now repopulated correctly - fix verified",
        "root_cause": "Input text not persisted in DB on save operation",
        "resolution": "Updated save logic; applied DB migration", "date": "21-04-2026",
    },
    {
        "tc_id": "TC_011", "version": "V1.2", "sprint": "Sprint 4",
        "feature": "AI Model Selection", "bug_id": "BUG_009",
        "title": "No Default Model Set on Page Load - FIXED",
        "status": "PASS",
        "result_line": "Default model BART preselected correctly on page load",
        "terminal_lines": [
            ("prompt",  "$ python app.py --test default_model"),
            ("normal",  "Loading summariser homepage (fresh session)..."),
            ("warn",    "BUG was: Selector shows BART but model_id = None in state"),
            ("normal",  "Applying fix: set DEFAULT_MODEL=BART in init script"),
            ("normal",  "Re-testing page load..."),
            ("normal",  "model_id on load: bart"),
            ("bar",     "[====================] 100%  1/1 model checks passed"),
            ("success", "SUCCESS: BART preselected and confirmed as active model"),
        ],
        "expected":   "BART preselected as default model on page load",
        "actual":     "BART now correctly preselected from init config - fix verified",
        "root_cause": "DEFAULT_MODEL config missing from front-end init script",
        "resolution": "Added DEFAULT_MODEL=BART to front-end init config", "date": "21-04-2026",
    },
    {
        "tc_id": "TC_012", "version": "V1.2", "sprint": "Sprint 4",
        "feature": "AI Model Selection", "bug_id": "BUG_010",
        "title": "Selected Model Not Sent to API - FIXED",
        "status": "PASS",
        "result_line": "API response confirms T5 was used for summarisation",
        "terminal_lines": [
            ("prompt",  "$ python app.py --test model_api_param"),
            ("normal",  "Selecting T5 from model dropdown..."),
            ("normal",  "Clicking Summarise..."),
            ("warn",    "BUG was: POST /api/summarise missing 'model_id' field"),
            ("normal",  "Applying fix: include model_id in API request payload"),
            ("normal",  "POST /api/summarise  payload: {model_id: 't5', ...}"),
            ("bar",     "[====================] 100%  metadata model = T5"),
            ("success", "SUCCESS: Response metadata confirms T5 was used"),
        ],
        "expected":   "Response confirms T5 was used for summarisation",
        "actual":     "API now receives model_id correctly - fix verified",
        "root_cause": "model_id param not included in API request payload",
        "resolution": "Updated API call to include model_id in payload", "date": "21-04-2026",
    },
    {
        "tc_id": "TC_013", "version": "V1.2", "sprint": "Sprint 4",
        "feature": "AI Model Selection", "bug_id": "BUG_011",
        "title": "Switching Model Does Not Change Output - FIXED",
        "status": "PASS",
        "result_line": "T5 and BART outputs differ: model switch confirmed",
        "terminal_lines": [
            ("prompt",  "$ python app.py --test model_switch_output"),
            ("normal",  "Summarising 800-word text with BART..."),
            ("normal",  "BART output length: 142 words"),
            ("normal",  "Switching model to T5 and re-summarising..."),
            ("normal",  "T5 output length: 117 words  [DIFFERENT from BART]"),
            ("warn",    "BUG was: Same root cause as TC_012 -- model_id not in payload"),
            ("normal",  "Same fix applied: model_id now included in payload"),
            ("bar",     "[====================] 100%  outputs differ = PASS"),
            ("success", "SUCCESS: Model switch produces distinct output - confirmed"),
        ],
        "expected":   "T5 output differs from BART output; confirms model switch",
        "actual":     "Outputs now differ correctly after payload fix - verified",
        "root_cause": "Same as TC_012: model_id missing from API payload",
        "resolution": "Same fix as TC_012: model_id added to payload", "date": "21-04-2026",
    },
]

FAIL_RECORDS = [
    {
        "tc_id": "TC_001", "version": "V1.2", "sprint": "Sprint 1",
        "feature": "Text Input (US1)", "bug_id": "BUG_001",
        "title": "Word Count Not Updating - BUG FOUND",
        "status": "FAIL",
        "result_line": "FAIL: Word count stuck at 0 after paste -- counter not firing",
        "terminal_lines": [
            ("prompt",  "$ python app.py --test word_count"),
            ("normal",  "Initialising summariser UI..."),
            ("normal",  "Pasting 500-word paragraph into textarea..."),
            ("normal",  "Reading word-count label from DOM..."),
            ("fail",    "ASSERT FAILED: expected=500  actual=0"),
            ("error",   "ERROR: onInput listener not bound -- counter never fires"),
            ("bar_fail","[====================] FAILED  0/3 checks passed"),
            ("fail",    "FAIL: Word count NOT updated -- Bug logged as BUG_001"),
        ],
        "expected":   "Word count updates in real time after paste",
        "actual":     "Word count stays at 0 -- counter does not update after paste",
        "root_cause": "onInput event listener not bound to textarea on initial render",
        "resolution": "Open -- initial run", "date": "17-04-2026",
    },
    {
        "tc_id": "TC_003", "version": "V1.2", "sprint": "Sprint 1",
        "feature": "Text Input (US1)", "bug_id": "BUG_002",
        "title": "Empty Input Warning Not Shown - BUG FOUND",
        "status": "FAIL",
        "result_line": "FAIL: No warning shown -- summariser attempted with empty input",
        "terminal_lines": [
            ("prompt",  "$ python app.py --test empty_input_validation"),
            ("normal",  "Launching summariser homepage..."),
            ("normal",  "Leaving input area empty..."),
            ("normal",  "Clicking Summarise button..."),
            ("normal",  "Checking for warning dialog / toast..."),
            ("fail",    "ASSERT FAILED: expected warning message -- got None"),
            ("error",   "ERROR: Backend received null payload -- no validation guard"),
            ("bar_fail","[====================] FAILED  0/2 checks passed"),
            ("fail",    "FAIL: Empty input silently sent to API -- Bug BUG_002"),
        ],
        "expected":   "Warning message: 'Please enter text to summarise'",
        "actual":     "No warning shown -- API called with empty/null payload",
        "root_cause": "Frontend validation skipped; backend receives null payload",
        "resolution": "Open -- initial run", "date": "17-04-2026",
    },
    {
        "tc_id": "TC_004", "version": "V1.2", "sprint": "Sprint 1",
        "feature": "Text Input (API)", "bug_id": "BUG_003",
        "title": "Word-Count Label Resets on Small Values - BUG FOUND",
        "status": "FAIL",
        "result_line": "FAIL: Word count label reset to 0 for short text inputs",
        "terminal_lines": [
            ("prompt",  "$ python app.py --test word_count_edge_case"),
            ("normal",  "Loading input handler..."),
            ("normal",  "Entering short text: 'Hello world' (2 words)..."),
            ("normal",  "Reading word-count label..."),
            ("fail",    "ASSERT FAILED: expected=2  actual=0"),
            ("error",   "ERROR: int guard resets label when value < threshold"),
            ("normal",  "Testing 5-word input..."),
            ("fail",    "ASSERT FAILED: expected=5  actual=0"),
            ("bar_fail","[====================] FAILED  0/3 edge cases passed"),
            ("fail",    "FAIL: All small-value inputs reset counter -- Bug BUG_003"),
        ],
        "expected":   "Word count label shows correctly for any input size",
        "actual":     "Label reset to 0 for all inputs with fewer than 10 words",
        "root_cause": "Edge-case guard using small integer values resets label to zero",
        "resolution": "Open -- initial run", "date": "17-04-2026",
    },
    {
        "tc_id": "TC_006", "version": "V1.2", "sprint": "Sprint 2",
        "feature": "Copy to Clipboard (US6)", "bug_id": "BUG_004",
        "title": "Copy Button Still Clickable When Disabled - BUG FOUND",
        "status": "FAIL",
        "result_line": "FAIL: Copy button fired click event despite greyed appearance",
        "terminal_lines": [
            ("prompt",  "$ python app.py --test copy_button_state"),
            ("normal",  "Loading homepage -- no summary generated..."),
            ("normal",  "Inspecting Copy button attributes..."),
            ("fail",    "ASSERT FAILED: button.disabled should be true -- got false"),
            ("normal",  "Simulating click on Copy button..."),
            ("error",   "ERROR: click event fired -- copy attempted with no content"),
            ("fail",    "ASSERT FAILED: pointer-events:none not applied to button"),
            ("bar_fail","[====================] FAILED  0/2 checks passed"),
            ("fail",    "FAIL: Disabled button remains interactive -- Bug BUG_004"),
        ],
        "expected":   "Copy button greyed / disabled and non-interactive before summary",
        "actual":     "Button visually greyed but click events still fire",
        "root_cause": "Copy button visually disabled but remained interactive",
        "resolution": "Open -- initial run", "date": "17-04-2026",
    },
    {
        "tc_id": "TC_007", "version": "V1.2", "sprint": "Sprint 2",
        "feature": "Copy to Clipboard (US6)", "bug_id": "BUG_005",
        "title": "Clipboard API Blocked -- Copy Failed - BUG FOUND",
        "status": "FAIL",
        "result_line": "FAIL: navigator.clipboard.writeText() blocked -- paste fails",
        "terminal_lines": [
            ("prompt",  "$ python app.py --test clipboard_copy"),
            ("normal",  "Generating summary text..."),
            ("normal",  "Clicking Copy button..."),
            ("error",   "DOMException: navigator.clipboard -- NotAllowedError"),
            ("error",   "ERROR: Clipboard API blocked in non-HTTPS context"),
            ("normal",  "Attempting paste in new tab..."),
            ("fail",    "ASSERT FAILED: clipboard = old content (not summary text)"),
            ("bar_fail","[====================] FAILED  0/1 paste verification"),
            ("fail",    "FAIL: Summary NOT copied to clipboard -- Bug BUG_005"),
        ],
        "expected":   "Summary text copied to clipboard; paste reproduces the text",
        "actual":     "Clipboard write blocked -- paste reproduces old clipboard content",
        "root_cause": "Clipboard API write blocked in non-HTTPS context; no fallback",
        "resolution": "Open -- initial run", "date": "19-04-2026",
    },
    {
        "tc_id": "TC_008", "version": "V1.2", "sprint": "Sprint 2",
        "feature": "Copy to Clipboard (US6)", "bug_id": "BUG_006",
        "title": "Toast Notification Missing After Copy - BUG FOUND",
        "status": "FAIL",
        "result_line": "FAIL: No toast notification shown after successful copy",
        "terminal_lines": [
            ("prompt",  "$ python app.py --test toast_notification"),
            ("normal",  "Generating summary and clicking Copy..."),
            ("normal",  "Clipboard write: OK"),
            ("normal",  "Checking toast component visibility..."),
            ("fail",    "ASSERT FAILED: toast.visible expected True -- got False"),
            ("error",   "ERROR: copy callback does not set toast.visible=True"),
            ("normal",  "Waiting 3s for delayed render..."),
            ("fail",    "ASSERT FAILED: toast still not visible after 3s"),
            ("bar_fail","[====================] FAILED  0/1 toast checks"),
            ("fail",    "FAIL: Toast notification never appears -- Bug BUG_006"),
        ],
        "expected":   "Toast notification appears for 3 seconds then disappears",
        "actual":     "No toast shown at all after copy -- UI gives no feedback",
        "root_cause": "Toast visibility flag not set in copy success callback",
        "resolution": "Open -- initial run", "date": "19-04-2026",
    },
    {
        "tc_id": "TC_009", "version": "V1.2", "sprint": "Sprint 3",
        "feature": "Usage History (US5)", "bug_id": "BUG_007",
        "title": "History Not Loading - BUG FOUND",
        "status": "FAIL",
        "result_line": "FAIL: 401 Unauthorized -- history list empty on load",
        "terminal_lines": [
            ("prompt",  "$ python app.py --test history_load"),
            ("normal",  "Logging in with test account (5 prior summaries)..."),
            ("normal",  "Navigating to History tab..."),
            ("normal",  "GET /api/history ..."),
            ("error",   "HTTP 401 Unauthorized -- token: <stale>"),
            ("fail",    "ASSERT FAILED: history list has 0 items -- expected 5"),
            ("error",   "ERROR: History API called before auth token refresh"),
            ("bar_fail","[====================] FAILED  0/5 history items loaded"),
            ("fail",    "FAIL: History panel empty -- stale token rejected -- BUG_007"),
        ],
        "expected":   "History list shows 5 summaries with timestamps",
        "actual":     "401 Unauthorized -- history API call rejected; list empty",
        "root_cause": "History API called before auth token refresh; stale token used",
        "resolution": "Open -- initial run", "date": "19-04-2026",
    },
    {
        "tc_id": "TC_010", "version": "V1.2", "sprint": "Sprint 3",
        "feature": "Usage History (US5)", "bug_id": "BUG_008",
        "title": "History Click Does Not Restore Input - BUG FOUND",
        "status": "FAIL",
        "result_line": "FAIL: Input box empty after history item selected",
        "terminal_lines": [
            ("prompt",  "$ python app.py --test history_repopulate"),
            ("normal",  "Opening history list and clicking saved entry..."),
            ("normal",  "Checking output box: populated [OK]"),
            ("normal",  "Checking input box..."),
            ("fail",    "ASSERT FAILED: input box is empty -- expected saved text"),
            ("error",   "ERROR: input_text column missing from history DB schema"),
            ("error",   "ERROR: Save handler never persisted input text to DB"),
            ("bar_fail","[====================] FAILED  1/2 fields repopulated"),
            ("fail",    "FAIL: Input not restored from history -- Bug BUG_008"),
        ],
        "expected":   "Input and output boxes repopulated with saved data",
        "actual":     "Output restored but input box remains empty",
        "root_cause": "Input text not persisted in DB on save operation",
        "resolution": "Open -- initial run", "date": "21-04-2026",
    },
    {
        "tc_id": "TC_011", "version": "V1.2", "sprint": "Sprint 4",
        "feature": "AI Model Selection", "bug_id": "BUG_009",
        "title": "No Default Model Set on Load - BUG FOUND",
        "status": "FAIL",
        "result_line": "FAIL: model_id=None in state despite BART shown in dropdown",
        "terminal_lines": [
            ("prompt",  "$ python app.py --test default_model"),
            ("normal",  "Loading summariser homepage (fresh session)..."),
            ("normal",  "Reading model selector value from DOM: BART"),
            ("normal",  "Reading model_id from app state..."),
            ("fail",    "ASSERT FAILED: state.model_id = None -- expected 'bart'"),
            ("normal",  "Clicking Summarise with no manual model selection..."),
            ("error",   "ERROR: Summarisation used T5-Base (fallback) not BART"),
            ("bar_fail","[====================] FAILED  0/1 model checks"),
            ("fail",    "FAIL: Default model not initialised properly -- BUG_009"),
        ],
        "expected":   "BART preselected as default model on page load",
        "actual":     "UI shows BART but state is None -- wrong model used for summary",
        "root_cause": "DEFAULT_MODEL config missing from front-end init script",
        "resolution": "Open -- initial run", "date": "21-04-2026",
    },
    {
        "tc_id": "TC_012", "version": "V1.2", "sprint": "Sprint 4",
        "feature": "AI Model Selection", "bug_id": "BUG_010",
        "title": "Selected Model Ignored by API - BUG FOUND",
        "status": "FAIL",
        "result_line": "FAIL: API response shows BART used -- T5 was selected",
        "terminal_lines": [
            ("prompt",  "$ python app.py --test model_api_param"),
            ("normal",  "Selecting T5 from model dropdown..."),
            ("normal",  "Clicking Summarise..."),
            ("normal",  "Inspecting POST /api/summarise payload..."),
            ("fail",    "ASSERT FAILED: 'model_id' key missing from request payload"),
            ("error",   "ERROR: API defaulted to BART -- model_id not forwarded"),
            ("fail",    "ASSERT FAILED: response.model_used=bart -- expected t5"),
            ("bar_fail","[====================] FAILED  0/1 model param checks"),
            ("fail",    "FAIL: Model selection ignored by backend -- Bug BUG_010"),
        ],
        "expected":   "Response confirms T5 was used for summarisation",
        "actual":     "Response shows BART used regardless of UI selection",
        "root_cause": "model_id param not included in API request payload",
        "resolution": "Open -- initial run", "date": "21-04-2026",
    },
    {
        "tc_id": "TC_013", "version": "V1.2", "sprint": "Sprint 4",
        "feature": "AI Model Selection", "bug_id": "BUG_011",
        "title": "Switching Model Does Not Change Output - BUG FOUND",
        "status": "FAIL",
        "result_line": "FAIL: Same summary returned after switching from BART to T5",
        "terminal_lines": [
            ("prompt",  "$ python app.py --test model_switch_output"),
            ("normal",  "Summarising 800-word text with BART..."),
            ("normal",  "BART output hash: a3f9c821"),
            ("normal",  "Switching model to T5 and re-summarising..."),
            ("normal",  "T5 output hash: a3f9c821"),
            ("fail",    "ASSERT FAILED: hashes match -- outputs are IDENTICAL"),
            ("error",   "ERROR: Same root cause as TC_012 -- model_id not in payload"),
            ("error",   "ERROR: API uses BART regardless of UI selection"),
            ("bar_fail","[====================] FAILED  outputs identical = FAIL"),
            ("fail",    "FAIL: Model switch has no effect on output -- Bug BUG_011"),
        ],
        "expected":   "T5 output differs from BART output; confirms model switch",
        "actual":     "Identical output hash -- API ignores model selection",
        "root_cause": "Same as TC_012: model_id missing from API payload",
        "resolution": "Open -- initial run", "date": "21-04-2026",
    },
]

PRIORITIES    = ["High","High","High","High","Low",
                 "Medium","High","Medium","High",
                 "Medium","High","High","High"]
SEVERITIES    = ["Major","Minor","Major","Major","Minor",
                 "Normal","Critical","Normal","Critical",
                 "Normal","Major","Critical","Major"]
INIT_STATUSES = ["FAIL","PASS","FAIL","FAIL","PASS",
                 "FAIL","FAIL","FAIL","FAIL",
                 "FAIL","FAIL","FAIL","FAIL"]

# ═════════════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _rect(ax, x, y, w, h, color, zorder=1):
    ax.add_patch(plt.Rectangle((x, y), w, h,
                                facecolor=color, edgecolor="none",
                                transform=ax.transData, zorder=zorder))

def _text(ax, x, y, txt, color, size, font=MONO, weight="normal",
          ha="left", va="top", zorder=2):
    ax.text(x, y, txt, color=color, fontsize=size,
            fontfamily=font, fontweight=weight,
            ha=ha, va=va, zorder=zorder, transform=ax.transData)

COLOR_MAP = {
    "prompt":   C_MONO_GREEN,
    "normal":   C_MONO_WHITE,
    "warn":     C_ORANGE_FAIL,
    "error":    C_MONO_RED,
    "fail":     C_MONO_RED,
    "success":  C_GREEN_PASS,
    "bar":      C_MONO_YELLOW,
    "bar_fail": "#ff4444",
}

# ═════════════════════════════════════════════════════════════════════════════
#  CARD GENERATOR
# ═════════════════════════════════════════════════════════════════════════════

def draw_card(rec: dict, idx: int, out_dir: str, total: int):
    W, H = 14.0, 8.8
    fig = plt.figure(figsize=(W, H), facecolor=C_BLACK)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.patch.set_facecolor(C_BLACK)

    is_pass   = rec["status"].upper() == "PASS"
    hdr_col   = C_GREEN_HDR  if is_pass else C_RED_HDR
    stat_bg   = C_GREEN_PASS if is_pass else C_RED_FAIL
    stat_txt  = "PASS"        if is_pass else "FAIL"
    rl_bg     = "#002200"     if is_pass else C_DARK_RED
    rl_col    = C_GREEN_PASS  if is_pass else C_MONO_RED
    strip_bg  = C_DARK_GREEN  if is_pass else C_DARK_RED

    HDR_H = 1.05
    _rect(ax, 0, H - HDR_H, W, HDR_H, hdr_col)

    # Title
    _text(ax, 0.22, H - 0.18, rec["title"], "white", 13.5, SANS, "bold")

    # PASS / FAIL badge
    bw, bh = 1.30, 0.40
    bx, by = W - bw - 0.18, H - 0.60
    _rect(ax, bx, by, bw, bh, stat_bg, zorder=3)
    _text(ax, bx + bw/2, by + bh/2 + 0.01, stat_txt,
          "black", 11, SANS, "bold", ha="center", va="center")

    # Meta
    my = H - HDR_H + 0.40
    _text(ax, 0.22, my,        f"Feature : {rec['feature']}", C_MONO_WHITE, 8)
    _text(ax, 0.22, my - 0.28, f"Version : {rec['version']}",  C_MONO_WHITE, 8)
    _text(ax, 0.22, my - 0.56, f"Sprint  : {rec['sprint']}",   C_MONO_WHITE, 8)
    _text(ax, W - 0.22, my, rec["bug_id"], C_META_GOLD, 9, weight="bold", ha="right")

    # Result summary strip
    rl_y = H - HDR_H - 0.01
    _rect(ax, 0, rl_y - 0.38, W, 0.38, rl_bg)
    _text(ax, 0.18, rl_y - 0.06,
          f">> {rec['result_line']}", rl_col, 9.5, weight="bold")

    # Terminal area
    TERM_TOP = rl_y - 0.38
    TERM_BOT = 1.30
    _rect(ax, 0, TERM_BOT, W, TERM_TOP - TERM_BOT, C_BLACK)

    # "TERMINAL OUTPUT" strip
    _rect(ax, 0, TERM_TOP - 0.01, W, -0.28, strip_bg)
    _text(ax, 0.18, TERM_TOP - 0.24, "TERMINAL OUTPUT", C_MONO_GREY, 7)

    # Lines
    ly = TERM_TOP - 0.52
    for kind, txt in rec["terminal_lines"]:
        if ly < TERM_BOT + 0.10:
            break
        _text(ax, 0.30, ly, txt, COLOR_MAP.get(kind, C_MONO_WHITE), 8.0)
        ly -= 0.44

    # Footer
    _rect(ax, 0, 0, W, TERM_BOT, "#111111")
    ax.axhline(TERM_BOT, color="#333333", linewidth=1, zorder=2)

    act_col = C_MONO_RED if not is_pass else C_GREEN_PASS

    _text(ax, 0.22, TERM_BOT - 0.08,  "EXPECTED  :", C_MONO_GREY, 7.5)
    _text(ax, 1.45, TERM_BOT - 0.08,  rec["expected"],   C_GREEN_PASS, 7.5)
    _text(ax, 0.22, TERM_BOT - 0.42,  "ACTUAL    :", C_MONO_GREY, 7.5)
    _text(ax, 1.45, TERM_BOT - 0.42,  rec["actual"],     act_col,     7.5)
    _text(ax, 0.22, TERM_BOT - 0.76,  "ROOT CAUSE:", C_MONO_GREY, 7.5)
    _text(ax, 1.45, TERM_BOT - 0.76,  rec["root_cause"], C_MONO_YELLOW, 7.5)

    _text(ax, W - 0.22, TERM_BOT - 0.08,
          f"Resolution : {rec['resolution']}", C_MONO_GREY, 7.5, ha="right")
    _text(ax, W - 0.22, TERM_BOT - 0.42,
          f"Date       : {rec['date']}",       C_MONO_GREY, 7.5, ha="right")
    _text(ax, W - 0.22, TERM_BOT - 0.76,
          f"Test Case  : {rec['tc_id']}",      C_MONO_GREY, 7.5, ha="right")

    fname = f"{rec['tc_id']}_{rec['status'].lower()}_bug_report.png"
    path  = os.path.join(out_dir, fname)
    plt.savefig(path, bbox_inches="tight", facecolor=C_BLACK, dpi=160)
    plt.close(fig)
    print(f"    [{idx:02d}/{total}] {fname}")


# ═════════════════════════════════════════════════════════════════════════════
#  OVERVIEW CARD
# ═════════════════════════════════════════════════════════════════════════════

def draw_overview():
    W, H = 20.0, 13.0
    fig = plt.figure(figsize=(W, H), facecolor=C_BLACK)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    fig.patch.set_facecolor(C_BLACK)

    _rect(ax, 0, H - 0.90, W, 0.90, C_GREEN_HDR)
    _text(ax, W/2, H - 0.27,
          "Context-Based Text Summarizer  |  QA Bug Report Overview",
          "white", 14, SANS, "bold", ha="center")
    _text(ax, W/2, H - 0.65,
          "All 13 Test Cases  -  Sprint V1.2  -  April 2026",
          C_MONO_YELLOW, 9, ha="center")

    ch_y = H - 0.92
    _rect(ax, 0, ch_y - 0.42, W, 0.42, "#002200")
    cols_x   = [0.20, 1.55, 3.80, 6.90, 9.50, 12.20, 14.80, 17.00, 18.80]
    col_hdrs = ["TC #","Feature","Bug Title (abbreviated)","Bug ID",
                "Priority","Severity","Init Status","Retry","Date"]
    for cx, ch in zip(cols_x, col_hdrs):
        _text(ax, cx, ch_y - 0.08, ch, C_MONO_GREY, 8)

    PRIC = {"High":C_MONO_RED,"Medium":C_ORANGE_FAIL,"Low":C_GREEN_PASS}
    SEVC = {"Critical":"#cc44ff","Major":C_MONO_RED,"Normal":"#4488ff","Minor":C_GREEN_PASS}

    row_y = ch_y - 0.42
    row_h = 0.62
    for i, rec in enumerate(PASS_RECORDS):
        bg = "#111111" if i % 2 == 0 else "#161616"
        _rect(ax, 0, row_y - row_h, W, row_h, bg)

        init_s = INIT_STATUSES[i]
        sc  = C_GREEN_PASS if init_s == "PASS" else C_MONO_RED
        pri = PRIORITIES[i]; sev = SEVERITIES[i]
        short = (rec["title"][:38] + "..") if len(rec["title"]) > 38 else rec["title"]
        ty = row_y - 0.28

        _text(ax, cols_x[0], ty, rec["tc_id"],         "#4af",       8.5, weight="bold")
        _text(ax, cols_x[1], ty, rec["feature"][:18],  C_MONO_WHITE, 7.5)
        _text(ax, cols_x[2], ty, short,                C_MONO_WHITE, 7.5)
        _text(ax, cols_x[3], ty, rec["bug_id"],        C_META_GOLD,  7.5)
        _text(ax, cols_x[4], ty, f"[{pri}]", PRIC.get(pri, C_MONO_WHITE), 7.5)
        _text(ax, cols_x[5], ty, f"[{sev}]", SEVC.get(sev, C_MONO_WHITE), 7.5)
        _text(ax, cols_x[6], ty, f"[ {init_s} ]",     sc,           8.5, weight="bold")
        _text(ax, cols_x[7], ty, "PASS",               C_GREEN_PASS, 8.5, weight="bold")
        _text(ax, cols_x[8], ty, rec["date"],          C_MONO_GREY,  7)

        row_y -= row_h

    pf = sum(1 for s in INIT_STATUSES if s == "PASS")
    ff = len(INIT_STATUSES) - pf
    _rect(ax, 0, 0, W, 0.72, "#002200")
    _text(ax, W/2, 0.48,
          f"TOTAL: {len(INIT_STATUSES)}   |   INITIAL PASS: {pf}   |   INITIAL FAIL: {ff}"
          "   |   RETRY PASS RATE: 100.0%",
          C_GREEN_PASS, 10, weight="bold", ha="center")

    path = os.path.join(OUT_DIR, "TC_000_overview.png")
    plt.savefig(path, bbox_inches="tight", facecolor=C_BLACK, dpi=160)
    plt.close(fig)
    print(f"    [00] TC_000_overview.png  ->  bug_reports/")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 64)
    print("  Terminal-Style Bug Report Image Generator")
    print("  Context-Based Text Summarizer  |  QA Sprint V1.2")
    print("=" * 64)

    print("\n  [OVERVIEW]")
    draw_overview()

    print(f"\n  [PASS set -- fixed state]  ->  bug_reports/pass/")
    for i, rec in enumerate(PASS_RECORDS, 1):
        draw_card(rec, i, PASS_DIR, len(PASS_RECORDS))

    print(f"\n  [FAIL set -- bug-found state]  ->  bug_reports/fail/")
    for i, rec in enumerate(FAIL_RECORDS, 1):
        draw_card(rec, i, FAIL_DIR, len(FAIL_RECORDS))

    nP = len(PASS_RECORDS) + 1
    nF = len(FAIL_RECORDS)
    print(f"\n{'=' * 64}")
    print(f"  Done!  {nP} PASS images  +  {nF} FAIL images  +  1 overview")
    print(f"  bug_reports/pass/   --  {nP} files")
    print(f"  bug_reports/fail/   --  {nF} files")
    print(f"  bug_reports/        --  1 overview")
    print(f"  (0 existing project files modified)")
    print("=" * 64)
