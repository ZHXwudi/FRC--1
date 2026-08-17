from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_app_renders_without_exception() -> None:
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30)
    app.run()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "平衡重建",
        "脉冲诊断",
        "敏感性",
        "同步代理模型",
        "研究证据",
        "智能体架构",
        "面试表达",
    ]
