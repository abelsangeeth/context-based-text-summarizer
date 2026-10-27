"""Run every Streamlit page headlessly (no model is loaded until a button is pressed)."""
import os

import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES = sorted(f"pages/{p}" for p in os.listdir(os.path.join(ROOT, "pages")) if p.endswith(".py"))


@pytest.fixture(autouse=True)
def _repo_cwd(monkeypatch):
    monkeypatch.chdir(ROOT)


def _open(page=None):
    # Start from app.py like `streamlit run app.py` does, so the sidebar's
    # st.page_link targets resolve, then navigate to the page under test.
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=180).run()
    if page:
        at.switch_page(page).run()
    return at


def test_home_page_renders():
    at = _open()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_error(page):
    at = _open(page)
    assert not at.exception, [e.value for e in at.exception]


def test_multilingual_page_accepts_chinese_text():
    at = _open("pages/5_Multilingual.py")
    at.text_area[0].input(
        "人工智能正在迅速改变全球各个行业。在医疗领域，人工智能诊断工具能够更早地发现疾病，"
        "显著改善患者的治疗效果。在金融领域，机器学习算法被用于欺诈检测和风险评估。"
    ).run()
    assert not at.exception
    assert any("Chinese" in m.value for m in at.markdown)
    summarize = next(b for b in at.button if b.label == "Summarize")
    assert not summarize.disabled
