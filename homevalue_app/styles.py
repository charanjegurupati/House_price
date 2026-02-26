import streamlit as st


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @keyframes fadeUp {
            0% { opacity: 0; transform: translateY(16px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            100% { background-position: 100% 50%; }
        }
        @keyframes floatY {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-8px); }
        }
        .hero {
            border-radius: 14px;
            padding: 14px 16px;
            color: #0f172a;
            border: 1px solid #bae6fd;
            background: linear-gradient(120deg, #e0f2fe, #f0fdfa, #ecfdf5);
            background-size: 240% 240%;
            animation: gradientShift 5s linear infinite alternate;
            margin-bottom: 12px;
        }
        .hero, .hero * {
            color: #0f172a !important;
        }
        .card {
            animation: fadeUp 700ms ease-out;
            border: 1px solid #d1fae5;
            border-radius: 14px;
            background: #ffffff;
            padding: 12px 14px;
            margin-top: 8px;
        }
        .card, .card * {
            color: #0f172a !important;
        }
        #code {
            font-family: 'Courier New', Courier, monospace;
            background: #f1f5f9;
            background-color: black;
            padding: 2px 6px;
            border-radius: 4px;
            border: 1px solid #e2e8f0;
            color: green !important;
}
        .welcome-center {
            text-align: center;
            font-size: 2rem;
            font-weight: 800;
            color: #14532d;
            margin-top: 22vh;
            margin-bottom: 22vh;
            letter-spacing: 0.6px;
        }
        .cursor {
            display: inline-block;
            animation: blink 0.8s step-end infinite;
        }
        @keyframes blink {
            50% { opacity: 0; }
        }
        .stImage img {
            animation: floatY 4s ease-in-out infinite;
        }
        section[data-testid="stSidebar"] div.stButton > button {
            width: 100%;
            border-radius: 10px;
            border: 1px solid #bae6fd;
            background: #f8fafc;
            color: #0f172a !important;
            font-weight: 700;
            transition: all 0.2s ease;
        }
        section[data-testid="stSidebar"] div.stButton > button:hover {
            transform: translateY(-1px);
            border-color: #38bdf8;
        }
        section[data-testid="stSidebar"] div.stButton > button[kind="secondary"] {
            background: #eff6ff;
            color: #0f172a !important;
        }
        section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #0ea5e9, #22c55e);
            color: #ffffff !important;
            border: none;
            box-shadow: 0 6px 16px rgba(14, 165, 233, 0.25);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
