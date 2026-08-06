"""
styles.py
---------
All custom CSS for the Diabetes Risk Predictor app lives here, kept
separate from app.py so the layout/logic code doesn't get cluttered
with styling. app.py just calls inject_global_css(accent) and this
file returns the <style> block as a string.
"""

import streamlit as st


def get_css(accent: str) -> str:
    """Return the full <style> block, with the current model's accent
    color baked in (used for buttons, header banner, etc.)."""
    return f"""
        <style>
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1100px;
        }}

        /* Buttons take the current model's accent color */
        .stButton > button {{
            background-color: {accent};
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.55rem 1.4rem;
            font-weight: 600;
            transition: filter 0.15s ease;
        }}
        .stButton > button:hover {{
            filter: brightness(1.08);
            color: white;
            border: none;
        }}

        /* Number inputs */
        div[data-baseweb="input"] {{
            border-radius: 8px;
        }}

        /* Section card */
        .section-card {{
            border: 1px solid rgba(150,150,150,0.25);
            border-radius: 14px;
            padding: 1.4rem 1.6rem 1.1rem 1.6rem;
            margin-bottom: 1.2rem;
            background: rgba(150,150,150,0.04);
        }}
        .section-card h4 {{
            margin-top: 0;
            margin-bottom: 1rem;
        }}

        /* Header banner */
        .app-header {{
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 1.1rem 1.5rem;
            border-radius: 14px;
            background: linear-gradient(135deg, {accent}22, {accent}08);
            border: 1px solid {accent}33;
            margin-bottom: 1.5rem;
        }}
        .app-header .icon {{
            font-size: 2.1rem;
        }}
        .app-header h2 {{
            margin: 0;
            color: {accent};
            font-weight: 800;
        }}
        .app-header p {{
            margin: 2px 0 0 0;
            color: rgba(200,200,200,0.85);
            font-size: 0.92rem;
        }}

        /* Result card */
        .result-card {{
            border-radius: 14px;
            padding: 1.4rem 1.6rem;
            margin-top: 0.6rem;
            border: 1px solid rgba(150,150,150,0.25);
            background: rgba(150,150,150,0.04);
        }}
        .result-title {{
            font-size: 0.85rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: rgba(200,200,200,0.7);
            margin-bottom: 6px;
        }}
        .result-label {{
            font-size: 1.6rem;
            font-weight: 800;
            margin-bottom: 10px;
        }}
        .risk-bar-track {{
            background: rgba(150,150,150,0.2);
            border-radius: 999px;
            height: 22px;
            width: 100%;
            overflow: hidden;
        }}
        .risk-bar-fill {{
            height: 100%;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 10px;
            color: white;
            font-weight: 700;
            font-size: 0.78rem;
        }}
        .disclaimer {{
            font-size: 0.8rem;
            color: rgba(180,180,180,0.7);
            margin-top: 0.6rem;
        }}
        </style>
    """


def inject_global_css(accent: str) -> None:
    """Render the CSS into the page. Call once per run, near the top."""
    st.markdown(get_css(accent), unsafe_allow_html=True)
