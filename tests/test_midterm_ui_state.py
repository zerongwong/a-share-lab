from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAGE = PROJECT_ROOT / "src" / "ashare_lab" / "ui" / "pages" / "08_中期主升组合.py"


def _source() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_submission_clears_previous_run_before_research_starts() -> None:
    source = _source()
    submitted = source.index("if submitted:")
    clear = source.index("st.session_state.pop(LATEST_MIDTERM_RUN_KEY, None)", submitted)
    research_try = source.index("    try:", submitted)
    save = source.index("st.session_state[LATEST_MIDTERM_RUN_KEY] = {", research_try)

    assert submitted < clear < research_try < save


def test_success_saves_one_atomic_midterm_run_payload() -> None:
    source = _source()

    assert 'LATEST_MIDTERM_RUN_KEY = "latest_midterm_run"' in source
    assert '"method_version": MIDTERM_METHOD_VERSION' in source
    assert 'st.session_state["latest_midterm_result"]' not in source
    assert 'st.session_state["latest_midterm_universe"]' not in source
    assert (
        "st.session_state[LATEST_MIDTERM_RUN_KEY] = {\n"
        '                "request": current_request,\n'
        '                "result": result,\n'
        '                "universe": universe,\n'
        "            }"
    ) in source


def test_rendering_requires_run_request_to_match_current_controls() -> None:
    source = _source()
    load = source.index("latest_run = st.session_state.get(LATEST_MIDTERM_RUN_KEY)")
    default_result = source.index("result: MidtermPortfolioResult | None = None", load)
    request_guard = source.index('latest_run.get("request") == current_request', default_result)
    expose_result = source.index('result = latest_run["result"]', request_guard)
    render = source.index("if result is not None:", expose_result)

    assert load < default_result < request_guard < expose_result < render
