"""Stitch-inspired Material Design 3 theme for ContextFree-X.

Call `apply_stitch_theme()` at the top of every page (after st.set_page_config)
to inject fonts, icons, and all CSS overrides.
"""

import streamlit as st

_FONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Manrope:wght@400;600;700;800"
    "&family=Inter:wght@400;500;600"
    "&display=swap"
)
_ICONS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Material+Symbols+Outlined:wght,FILL@100..700,0..1"
    "&display=swap"
)

_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="{fonts}" rel="stylesheet"/>
<link href="{icons}" rel="stylesheet"/>

<style>
/* ═══════════════════════════════════════════════
   GLOBALS
═══════════════════════════════════════════════ */
#MainMenu, footer {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}

/* Hide default Streamlit page navigation */
[data-testid="stSidebarNav"] {{ display: none !important; }}
[data-testid="stSidebarNavItems"] {{ display: none !important; }}
[data-testid="stSidebarNavSeparator"] {{ display: none !important; }}

body, .stApp {{ background-color: #f9f9ff !important; }}

body, p, label, .stMarkdown, .stText, .stCaption,
input, textarea, select, button {{
    font-family: 'Inter', sans-serif !important;
}}

h1, h2, h3, h4, h5, h6 {{
    font-family: 'Manrope', sans-serif !important;
    letter-spacing: -0.025em;
}}

.material-symbols-outlined {{
    font-family: 'Material Symbols Outlined' !important;
    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
    font-weight: normal;
    font-style: normal;
    vertical-align: middle;
    font-size: 20px;
    line-height: 1;
    display: inline-block;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}}

/* ═══════════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════════ */
section[data-testid="stSidebar"] {{
    background-color: #f8fafc !important;
    border-right: none !important;
    box-shadow: 1px 0 0 0 rgba(194, 198, 214, 0.35);
}}

section[data-testid="stSidebar"] > div:first-child {{
    padding-top: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
}}

/* Style st.page_link in sidebar — target every possible inner element */
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"],
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] > div,
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] > div > div {{
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    width: 100% !important;
    text-decoration: none !important;
    background: transparent !important;
    border: none !important;
}}

section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {{
    padding: 10px 16px !important;
    margin: 0 12px 3px 12px !important;
    border-radius: 12px !important;
    transition: background 0.15s !important;
}}

section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {{
    background: rgba(100, 116, 139, 0.09) !important;
}}

/* Force dark visible text on ALL text inside page_link */
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] p,
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] span,
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] div {{
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    color: #1e293b !important;
    margin: 0 !important;
    letter-spacing: -0.015em !important;
    white-space: nowrap !important;
    overflow: visible !important;
    opacity: 1 !important;
    visibility: visible !important;
}}

/* Keep emoji/icon span default rendering */
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] span[role="img"],
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] .material-symbols-outlined {{
    font-family: inherit !important;
    font-size: 1.1rem !important;
}}

/* Sidebar radio / checkbox labels */
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stCheckbox label {{
    padding: 0 16px !important;
}}

section[data-testid="stSidebar"] .stRadio > div,
section[data-testid="stSidebar"] .stCheckbox > label {{
    padding-left: 16px !important;
    padding-right: 16px !important;
}}

section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] label,
section[data-testid="stSidebar"] .stCheckbox label span {{
    color: #424754 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
}}

section[data-testid="stSidebar"] .stRadio > label,
section[data-testid="stSidebar"] .stCheckbox > label,
section[data-testid="stSidebar"] .stSelectbox > label {{
    padding-left: 16px !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    color: #191b23 !important;
}}

/* Sidebar section content padding */
section[data-testid="stSidebar"] .stMarkdown {{
    padding-left: 4px !important;
    padding-right: 4px !important;
}}

section[data-testid="stSidebar"] hr {{
    margin: 8px 16px !important;
}}

/* ── Sidebar brand ── */
.sb-brand {{
    padding: 28px 20px 20px 20px;
    border-bottom: 1px solid rgba(194, 198, 214, 0.25);
    margin-bottom: 12px;
}}

.sb-brand-name {{
    font-family: 'Manrope', sans-serif !important;
    font-size: 1.2rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: #0058be;
    line-height: 1;
    margin: 0 0 3px 0;
}}

.sb-brand-sub {{
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #727785;
    margin: 0;
}}

/* ── Nav items ── */
.sb-nav {{ padding: 0 12px; }}

.sb-nav-item {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 12px;
    border-radius: 12px;
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.875rem;
    font-weight: 600;
    letter-spacing: -0.015em;
    color: #64748b;
    margin-bottom: 3px;
    border-right: 3px solid transparent;
    transition: background 0.15s, color 0.15s;
    white-space: nowrap;
}}

.sb-nav-item:hover {{
    background: rgba(100, 116, 139, 0.09);
    color: #1e293b;
}}

.sb-nav-item.active {{
    background: #eff6ff;
    color: #1d4ed8;
    border-right: 3px solid #2563eb;
}}

.sb-nav-item .s-icon {{
    font-family: 'Material Symbols Outlined' !important;
    font-size: 20px;
    font-weight: 300;
    font-style: normal;
    line-height: 1;
    font-variation-settings: 'FILL' 0, 'wght' 300, 'GRAD' 0, 'opsz' 24;
    color: inherit;
}}

.sb-nav-item.active .s-icon {{
    font-variation-settings: 'FILL' 1, 'wght' 400, 'GRAD' 0, 'opsz' 24;
}}

/* ── Sidebar footer ── */
.sb-footer {{
    padding: 16px;
    margin-top: auto;
    border-top: 1px solid rgba(194, 198, 214, 0.25);
}}

.sb-footer-card {{
    background: #f2f3fd;
    border-radius: 12px;
    padding: 10px 14px;
    font-size: 11px;
    color: #424754;
    line-height: 1.5;
}}

.sb-footer-card strong {{
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #0058be;
    display: block;
    margin-bottom: 4px;
}}

/* ═══════════════════════════════════════════════
   MAIN AREA
═══════════════════════════════════════════════ */
.main .block-container {{
    padding-top: 2.5rem !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
    max-width: 1280px;
}}

/* ═══════════════════════════════════════════════
   BUTTONS
═══════════════════════════════════════════════ */
.stButton > button {{
    background: linear-gradient(135deg, #0058be 0%, #2170e4 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.875rem !important;
    padding: 10px 28px !important;
    box-shadow: 0 4px 14px rgba(0, 88, 190, 0.28) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    letter-spacing: -0.01em !important;
}}

.stButton > button:hover {{
    transform: scale(0.98) !important;
    box-shadow: 0 2px 8px rgba(0, 88, 190, 0.2) !important;
    color: white !important;
}}

.stButton > button:active {{
    transform: scale(0.96) !important;
}}

.stButton > button:disabled {{
    background: #e1e2ec !important;
    color: #a0a4b0 !important;
    box-shadow: none !important;
}}

/* ═══════════════════════════════════════════════
   INPUTS
═══════════════════════════════════════════════ */
.stTextArea textarea {{
    border-radius: 16px !important;
    border: 1px solid rgba(194, 198, 214, 0.55) !important;
    background: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.975rem !important;
    line-height: 1.75 !important;
    padding: 16px 20px !important;
    color: #191b23 !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.045) !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}}

.stTextArea textarea:focus {{
    border-color: rgba(0, 88, 190, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(0, 88, 190, 0.1) !important;
    outline: none !important;
}}

.stTextArea textarea::placeholder {{
    color: rgba(114, 119, 133, 0.55) !important;
}}

.stTextInput > div > div > input {{
    border-radius: 12px !important;
    border: 1px solid rgba(194, 198, 214, 0.55) !important;
    background: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    padding: 10px 16px !important;
    color: #191b23 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.045) !important;
}}

.stTextInput > div > div > input:focus {{
    border-color: rgba(0, 88, 190, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(0, 88, 190, 0.1) !important;
}}

/* ═══════════════════════════════════════════════
   SELECTBOX & MULTISELECT
═══════════════════════════════════════════════ */
.stSelectbox > div > div {{
    border-radius: 12px !important;
    border: 1px solid rgba(194, 198, 214, 0.55) !important;
    background: #ffffff !important;
}}

.stMultiSelect > div > div {{
    border-radius: 12px !important;
    border: 1px solid rgba(194, 198, 214, 0.55) !important;
    background: #ffffff !important;
}}

.stMultiSelect span[data-baseweb="tag"] {{
    background-color: #0058be !important;
    border-radius: 999px !important;
    color: white !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    padding: 4px 12px !important;
}}

/* ═══════════════════════════════════════════════
   METRIC CONTAINERS
═══════════════════════════════════════════════ */
div[data-testid="metric-container"] {{
    background: #ffffff !important;
    border-radius: 16px !important;
    border: 1px solid rgba(194, 198, 214, 0.28) !important;
    padding: 16px 20px !important;
    box-shadow: 0 1px 6px rgba(0, 0, 0, 0.05) !important;
}}

div[data-testid="metric-container"] label {{
    font-family: 'Inter', sans-serif !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #0058be !important;
}}

div[data-testid="metric-container"] [data-testid="stMetricValue"] div {{
    font-family: 'Manrope', sans-serif !important;
    font-size: 1.75rem !important;
    font-weight: 900 !important;
    color: #191b23 !important;
}}

/* ═══════════════════════════════════════════════
   FILE UPLOADER
═══════════════════════════════════════════════ */
section[data-testid="stFileUploader"] {{
    background: #ffffff !important;
    border-radius: 20px !important;
    border: 2px dashed rgba(194, 198, 214, 0.65) !important;
    padding: 28px !important;
    text-align: center !important;
    transition: border-color 0.2s !important;
}}

section[data-testid="stFileUploader"]:hover {{
    border-color: rgba(0, 88, 190, 0.45) !important;
}}

/* ═══════════════════════════════════════════════
   TABS
═══════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {{
    background: #f2f3fd !important;
    border-radius: 14px !important;
    padding: 4px !important;
    gap: 4px !important;
    border-bottom: none !important;
}}

.stTabs [data-baseweb="tab"] {{
    border-radius: 10px !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.875rem !important;
    padding: 8px 22px !important;
    color: #727785 !important;
    border: none !important;
    background: transparent !important;
    transition: all 0.15s !important;
}}

.stTabs [aria-selected="true"] {{
    background: #ffffff !important;
    color: #0058be !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.09) !important;
}}

.stTabs [data-baseweb="tab-highlight"] {{
    display: none !important;
}}

.stTabs [data-baseweb="tab-border"] {{
    display: none !important;
}}

/* ═══════════════════════════════════════════════
   DATA TABLES
═══════════════════════════════════════════════ */
.stDataFrame {{
    border-radius: 16px !important;
    border: 1px solid rgba(194, 198, 214, 0.25) !important;
    overflow: hidden !important;
}}

.stDataFrame thead tr th {{
    background: #f2f3fd !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #424754 !important;
    padding: 12px 16px !important;
}}

/* ═══════════════════════════════════════════════
   PROGRESS BAR
═══════════════════════════════════════════════ */
.stProgress > div > div > div > div {{
    background: linear-gradient(90deg, #0058be, #2170e4) !important;
    border-radius: 999px !important;
}}

.stProgress > div > div > div {{
    border-radius: 999px !important;
    background: #e1e2ec !important;
}}

/* ═══════════════════════════════════════════════
   EXPANDER
═══════════════════════════════════════════════ */
.streamlit-expanderHeader {{
    background: #f2f3fd !important;
    border-radius: 12px !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 700 !important;
}}

/* ═══════════════════════════════════════════════
   ALERTS / INFO BOXES
═══════════════════════════════════════════════ */
div[data-testid="stAlert"] {{
    border-radius: 14px !important;
    border-left-width: 4px !important;
    font-family: 'Inter', sans-serif !important;
}}

/* ═══════════════════════════════════════════════
   DIVIDER
═══════════════════════════════════════════════ */
hr {{
    border-color: rgba(194, 198, 214, 0.3) !important;
    margin: 20px 0 !important;
}}

/* ═══════════════════════════════════════════════
   SPINNER
═══════════════════════════════════════════════ */
.stSpinner > div {{
    border-color: #0058be transparent transparent transparent !important;
}}

/* ═══════════════════════════════════════════════
   RADIO BUTTONS
═══════════════════════════════════════════════ */
.stRadio [data-baseweb="radio"] {{
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}}

/* ═══════════════════════════════════════════════
   REUSABLE COMPONENT CLASSES
═══════════════════════════════════════════════ */

/* Page header */
.cfx-page-title {{
    font-family: 'Manrope', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: #191b23;
    margin: 0 0 4px 0;
    line-height: 1.15;
}}

.cfx-page-subtitle {{
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    color: #727785;
    line-height: 1.6;
    margin: 0 0 28px 0;
}}

.cfx-section-title {{
    font-family: 'Manrope', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #191b23;
    margin: 0 0 12px 0;
}}

/* Cards */
.cfx-card {{
    background: #ffffff;
    border-radius: 24px;
    padding: 28px 32px;
    border: 1px solid rgba(194, 198, 214, 0.2);
    box-shadow: 0 2px 16px rgba(25, 27, 35, 0.05);
}}

.cfx-card-sm {{
    background: #f2f3fd;
    border-radius: 16px;
    padding: 16px 20px;
    border: 1px solid rgba(194, 198, 214, 0.15);
}}

.cfx-output-card {{
    background: #ffffff;
    border-radius: 28px;
    padding: 36px 40px;
    border: 1px solid rgba(194, 198, 214, 0.18);
    box-shadow: 0 6px 28px rgba(25, 27, 35, 0.07);
    margin-top: 8px;
}}

/* Summary text */
.cfx-summary-text {{
    font-family: 'Inter', sans-serif;
    font-size: 1.02rem;
    line-height: 1.78;
    color: #424754;
}}

/* Metric pill */
.cfx-metric-pill {{
    background: #ffffff;
    border-radius: 16px;
    padding: 14px 18px;
    border: 1px solid rgba(194, 198, 214, 0.28);
    box-shadow: 0 1px 5px rgba(0,0,0,0.04);
}}

.cfx-metric-label {{
    font-family: 'Inter', sans-serif;
    font-size: 9px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #0058be;
    margin-bottom: 4px;
}}

.cfx-metric-value {{
    font-family: 'Manrope', sans-serif;
    font-size: 1.7rem;
    font-weight: 900;
    color: #191b23;
    line-height: 1;
}}

/* Badge */
.cfx-badge {{
    display: inline-flex;
    align-items: center;
    padding: 3px 10px;
    border-radius: 6px;
    font-family: 'Inter', sans-serif;
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    background: #e1e2ec;
    color: #405682;
}}

.cfx-badge-primary {{
    background: rgba(0, 88, 190, 0.1);
    color: #0058be;
}}

.cfx-badge-success {{
    background: rgba(16, 185, 129, 0.12);
    color: #065f46;
}}

/* Best model badge */
.cfx-best-badge {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 14px;
    border-radius: 999px;
    background: linear-gradient(135deg, #0058be, #2170e4);
    color: white;
    font-family: 'Inter', sans-serif;
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    box-shadow: 0 2px 8px rgba(0, 88, 190, 0.3);
}}

/* Model chip */
.cfx-model-chip {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 16px;
    border-radius: 999px;
    background: #0058be;
    color: white;
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    font-weight: 700;
    box-shadow: 0 2px 8px rgba(0, 88, 190, 0.28);
}}

/* Model comparison card */
.cfx-model-card {{
    background: #ffffff;
    border-radius: 20px;
    padding: 24px;
    border: 1px solid rgba(194, 198, 214, 0.22);
    box-shadow: 0 2px 12px rgba(25, 27, 35, 0.05);
    height: 100%;
    position: relative;
}}

.cfx-model-card-accent {{
    height: 4px;
    border-radius: 999px;
    margin-bottom: 20px;
}}

/* Score bar */
.cfx-score-bar-wrap {{
    background: #e1e2ec;
    border-radius: 999px;
    height: 6px;
    overflow: hidden;
    margin-top: 6px;
}}

.cfx-score-bar {{
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #0058be, #2170e4);
}}

/* Divider with label */
.cfx-divider {{
    display: flex;
    align-items: center;
    gap: 16px;
    margin: 28px 0;
}}

.cfx-divider::before,
.cfx-divider::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(194, 198, 214, 0.3);
}}

.cfx-divider-label {{
    font-family: 'Inter', sans-serif;
    font-size: 9px;
    font-weight: 900;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #727785;
}}

/* Tag/chip for detected languages, rooms, etc. */
.cfx-tag {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 8px;
    background: #e1e2ec;
    color: #191b23;
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    font-weight: 600;
    margin: 3px;
}}

.cfx-tag-primary {{
    background: rgba(0, 88, 190, 0.1);
    color: #0058be;
    border-left: 2px solid #0058be;
}}

/* Language detection pill */
.cfx-lang-pill {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 999px;
    background: rgba(0, 88, 190, 0.07);
    border: 1px solid rgba(0, 88, 190, 0.18);
    color: #0058be;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}

.cfx-lang-dot {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #0058be;
    animation: pulse 1.5s infinite;
}}

@keyframes pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.5; transform: scale(0.85); }}
}}

/* Upload drop zone */
.cfx-dropzone {{
    background: #ffffff;
    border-radius: 20px;
    border: 2px dashed rgba(194, 198, 214, 0.65);
    padding: 48px 32px;
    text-align: center;
    transition: border-color 0.2s;
}}

.cfx-dropzone:hover {{
    border-color: rgba(0, 88, 190, 0.45);
}}

.cfx-dropzone-icon {{
    font-family: 'Material Symbols Outlined' !important;
    font-size: 48px !important;
    color: #0058be;
    font-variation-settings: 'FILL' 0, 'wght' 300, 'GRAD' 0, 'opsz' 48;
    display: block;
    margin: 0 auto 16px;
    line-height: 1;
}}

/* Hero gradient text */
.cfx-gradient-text {{
    background: linear-gradient(135deg, #0058be 0%, #2170e4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

/* Input label */
.stTextArea label, .stTextInput label, .stSelectbox label,
.stMultiSelect label, .stFileUploader label {{
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
    color: #191b23 !important;
}}
</style>
"""


_HIDE_DEFAULT_NAV = """
<style>
[data-testid="stSidebarNav"] { display: none !important; }
[data-testid="stSidebarNavItems"] { display: none !important; }
[data-testid="stSidebarNavSeparator"] { display: none !important; }
</style>
"""


def apply_stitch_theme() -> None:
    """Inject Stitch-inspired CSS, Google Fonts, and Material Symbols into the page."""
    # Inject critical nav-hiding CSS first to prevent flash of default nav
    st.markdown(_HIDE_DEFAULT_NAV, unsafe_allow_html=True)
    st.markdown(
        _CSS.format(fonts=_FONTS_URL, icons=_ICONS_URL),
        unsafe_allow_html=True,
    )


# ── Sidebar helpers ──────────────────────────────────────────────────────────

def sidebar_brand() -> None:
    """Render the ContextFree-X brand block inside the sidebar."""
    st.markdown(
        """
        <div class="sb-brand">
            <div class="sb-brand-name">ContextFree-X</div>
            <div class="sb-brand-sub">Architectural Curator</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_nav(active: str) -> None:
    """Render themed navigation links using st.page_link. `active` is one of the page keys."""
    pages = [
        ("summarize",   "summarize",   "📝",  "Summarize",             "pages/1_Summarize.py"),
        ("compare",     "compare_arrows","⚖️", "Compare Models",        "pages/2_Compare_Models.py"),
        ("dashboard",   "dashboard",   "📊",  "Metrics Dashboard",     "pages/3_Metrics_Dashboard.py"),
        ("batch",       "layers",      "📋",  "Batch Evaluate",        "pages/4_Batch_Evaluate.py"),
        ("multilingual","language",    "🌐",  "Multilingual",          "pages/5_Multilingual.py"),
        ("blueprint",   "architecture","🏛️", "Blueprint Summarizer",  "pages/6_Blueprint_Summarizer.py"),
    ]

    for key, icon_name, emoji, label, path in pages:
        is_active = key == active
        if is_active:
            # Highlight active page with styled HTML (non-navigable since we're already on it)
            st.markdown(
                f"""<div style="display:flex;align-items:center;gap:10px;padding:10px 16px;
                     margin:0 12px 3px 12px;border-radius:12px;background:#eff6ff;color:#1d4ed8;
                     border-right:3px solid #2563eb;
                     font-family:'Manrope',sans-serif;font-size:0.875rem;font-weight:600;">
                     <span class="material-symbols-outlined"
                           style="font-size:20px;font-variation-settings:'FILL' 1,'wght' 400,'GRAD' 0,'opsz' 24;">
                       {icon_name}</span>{label}</div>""",
                unsafe_allow_html=True,
            )
        else:
            st.page_link(path, label=label, icon=emoji)


def sidebar_footer() -> None:
    """Render the bottom footer card inside the sidebar."""
    st.markdown(
        """
        <div class="sb-footer">
            <div class="sb-footer-card">
                <strong>Models</strong>
                T5 · BART · PEGASUS · mT5
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Page header helpers ──────────────────────────────────────────────────────

def page_header(title: str, subtitle: str = "") -> None:
    """Render a Stitch-styled page title + subtitle."""
    html = f'<div class="cfx-page-title">{title}</div>'
    if subtitle:
        html += f'<div class="cfx-page-subtitle">{subtitle}</div>'
    st.markdown(html, unsafe_allow_html=True)


def section_title(text: str) -> None:
    """Render a Stitch-styled section subheading."""
    st.markdown(f'<div class="cfx-section-title">{text}</div>', unsafe_allow_html=True)


# ── Inline component helpers ─────────────────────────────────────────────────

def metric_card(label: str, value: str) -> None:
    """Render a Stitch metric pill."""
    st.markdown(
        f"""<div class="cfx-metric-pill">
              <div class="cfx-metric-label">{label}</div>
              <div class="cfx-metric-value">{value}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def badge(text: str, variant: str = "") -> str:
    """Return an inline badge HTML string. variant: '' | 'primary' | 'success'"""
    cls = f"cfx-badge{'-' + variant if variant else ''}"
    return f'<span class="{cls}">{text}</span>'


def best_badge(text: str = "Best Model") -> str:
    """Return the golden 'Best Model' badge HTML string."""
    return (
        f'<span class="cfx-best-badge">'
        f'<span class="material-symbols-outlined" style="font-size:12px;">stars</span>'
        f'{text}</span>'
    )


def divider(label: str = "") -> None:
    """Render a styled horizontal rule with optional centred label."""
    inner = f'<span class="cfx-divider-label">{label}</span>' if label else ""
    st.markdown(f'<div class="cfx-divider">{inner}</div>', unsafe_allow_html=True)


def output_card(content_html: str) -> None:
    """Wrap arbitrary HTML in a Stitch output card."""
    st.markdown(
        f'<div class="cfx-output-card">{content_html}</div>',
        unsafe_allow_html=True,
    )


def icon(name: str, fill: int = 0, size: int = 20) -> str:
    """Return a Material Symbols icon span (inline HTML)."""
    return (
        f'<span class="material-symbols-outlined" '
        f'style="font-size:{size}px;'
        f"font-variation-settings:'FILL' {fill},'wght' 400,'GRAD' 0,'opsz' 24;\">"
        f"{name}</span>"
    )
