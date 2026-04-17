"""
gradio_compat.py
================
Gradio 버전 간 호환성 유틸리티

Gradio 버전별 theme 파라미터 위치 변화:
  - Gradio 3.x / 4.x / 5.x : gr.Blocks(theme=...)
  - Gradio 6.x+             : launch(theme=...)

Colab 환경 자동 처리:
  - share=True 자동 설정  (외부 터널 URL 발급)
  - inbrowser=False       (Colab에서 브라우저 자동실행 불필요)
  - server_name 제거      (Colab에서 0.0.0.0 바인딩 불필요)

사용법 (실습 파일 공통):
    from gradio_compat import make_blocks, safe_launch
"""

import gradio as gr


def get_gradio_major() -> int:
    """설치된 Gradio의 메이저 버전 번호 반환 (예: 5, 6)"""
    try:
        return int(gr.__version__.split(".")[0])
    except Exception:
        return 6  # 파싱 실패 시 현재 설치 버전(6.x) 기준으로 안전하게 처리


def make_blocks(title: str = "", theme=None):
    """
    Gradio 버전에 따라 theme 위치를 자동 조정하여 Blocks 객체 생성.

      - Gradio < 6 : gr.Blocks(title=..., theme=...)
      - Gradio >= 6: gr.Blocks(title=...)  ← theme은 launch()에서 전달

    사용 예:
        with make_blocks(title="RAG 실습", theme=gr.themes.Soft()) as demo:
            ...
    """
    if theme is None:
        return gr.Blocks(title=title)
    if get_gradio_major() >= 6:
        # Gradio 6+: Blocks 생성자에서 theme 파라미터 제거됨
        return gr.Blocks(title=title)
    else:
        # Gradio 3~5: Blocks 생성자에서 theme 전달
        return gr.Blocks(title=title, theme=theme)


def is_colab() -> bool:
    """Google Colab 환경 여부 확인"""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def safe_launch(demo, theme=None, server_port: int = 7860, **kwargs):
    """
    Gradio 버전 및 실행 환경(Colab / 로컬)에 따라 안전하게 launch() 호출.

    처리 내용:
      [Colab 환경 감지 시]
        - share=True  : Gradio 공개 터널 URL 자동 발급 → 브라우저에서 접속 가능
        - inbrowser   : 제거 (Colab 내부 브라우저 자동실행 불필요/불가)
        - server_name : 제거 (Colab에서 0.0.0.0 바인딩 불필요)

      [Gradio 버전 처리]
        - Gradio >= 6 : theme을 launch()에 전달
        - Gradio < 6  : theme은 이미 make_blocks()에서 Blocks()에 전달됨
                        → launch()에서 theme 인자 제거

    사용 예:
        safe_launch(
            demo,
            theme=gr.themes.Soft(),
            server_name="0.0.0.0",
            server_port=7860,
            share=False,       # Colab에서 자동으로 True로 전환됨
            inbrowser=True,    # Colab에서 자동으로 제거됨
        )
    """
    major = get_gradio_major()

    # ── Colab 환경 자동 보정 ──────────────────────────────
    if is_colab():
        kwargs["share"] = True           # 공개 URL 발급 (필수: Colab에서 UI 접속)
        kwargs.pop("inbrowser", None)    # Colab에서 불필요
        kwargs.pop("server_name", None)  # Colab에서 불필요

    # ── theme 처리 ────────────────────────────────────────
    if theme is not None and major >= 6:
        # Gradio 6+: launch()에서 theme 전달
        kwargs["theme"] = theme
    else:
        # Gradio < 6: theme은 Blocks()에서 이미 처리됨 → launch()에서 제거
        kwargs.pop("theme", None)

    demo.launch(server_port=server_port, **kwargs)
