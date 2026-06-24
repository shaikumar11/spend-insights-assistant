"""
Streamlit UI for the AI-Powered Customer Spend Insights Assistant (local/free version).
Styled to closely match the look and feel of Claude / ChatGPT: clean message flow,
no loud bubbles on the assistant side, a soft accent bubble for the user, and a
fixed, rounded input bar. Charts render inline with the answer they belong to,
each with a built-in download button. Includes a light/dark theme toggle.

Make sure Ollama is running first (ollama serve), then:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
from rag_pipeline import build_summary_documents, build_vector_store, answer_question, answer_question_stream
from chart_helper import maybe_build_chart

st.set_page_config(page_title="Spend Insights Assistant", page_icon="💳", layout="centered")

# ---------------------------------------------------------------------------
# Theme toggle
# ---------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

with st.sidebar:
    st.markdown("**Appearance**")
    theme_choice = st.radio(
        "Theme", ["Dark", "Light"],
        index=0 if st.session_state.theme == "dark" else 1,
        label_visibility="collapsed",
        horizontal=True,
    )
    st.session_state.theme = theme_choice.lower()

IS_DARK = st.session_state.theme == "dark"

if IS_DARK:
    BG = "#1a1a1a"
    BG_ELEVATED = "#222222"
    TEXT_PRIMARY = "#ececec"
    TEXT_SECONDARY = "#9b9b9b"
    BORDER = "#2e2e2e"
    USER_BUBBLE = "#2f3a4d"
    PLOTLY_TEMPLATE = "plotly_dark"
    GRID_COLOR = "rgba(255,255,255,0.08)"
else:
    BG = "#ffffff"
    BG_ELEVATED = "#f5f5f5"
    TEXT_PRIMARY = "#1a1a1a"
    TEXT_SECONDARY = "#6b6b6b"
    BORDER = "#e2e2e2"
    USER_BUBBLE = "#dbe6ff"
    PLOTLY_TEMPLATE = "plotly_white"
    GRID_COLOR = "rgba(0,0,0,0.08)"

ACCENT = "#5d8eff"

# Make charts theme-aware: chart_helper builds with a fixed dark template,
# so we re-theme the figure here based on the current toggle before rendering.
def themed(fig):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_PRIMARY),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR)
    return fig


# Plotly config: keeps the toolbar visible (download/zoom icons) and adds a
# clean filename for the downloaded PNG instead of Plotly's generic default.
PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"filename": "spend_insights_chart", "format": "png", "scale": 2},
}

st.markdown(f"""
<style>
    .stApp {{
        background-color: {BG};
    }}
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 7rem;
        max-width: 720px;
    }}
    html, body, [class*="css"] {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    }}

    .app-header {{
        display: flex;
        align-items: center;
        gap: 0.65rem;
        padding-bottom: 1.5rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid {BORDER};
    }}
    .app-header .mark {{
        width: 30px; height: 30px;
        border-radius: 8px;
        background: {ACCENT};
        display: flex; align-items: center; justify-content: center;
        font-size: 0.95rem;
        flex-shrink: 0;
    }}
    .app-header .name {{
        color: {TEXT_PRIMARY};
        font-size: 0.95rem;
        font-weight: 600;
        line-height: 1.3;
    }}
    .app-header .desc {{
        color: {TEXT_SECONDARY};
        font-size: 0.78rem;
        line-height: 1.3;
    }}

    .status-line {{
        display: flex; align-items: center; gap: 0.4rem;
        color: {TEXT_SECONDARY};
        font-size: 0.78rem;
        margin-bottom: 1.75rem;
    }}
    .status-line .dot {{
        width: 6px; height: 6px; border-radius: 50%;
        background: #3ecf6a;
        flex-shrink: 0;
    }}

    div[data-testid="stChatMessage"] {{
        background: transparent !important;
        padding: 0 !important;
        margin-bottom: 1.6rem !important;
        gap: 0.75rem !important;
    }}
    div[data-testid="stChatMessageAvatarUser"],
    div[data-testid="stChatMessageAvatarAssistant"] {{
        display: none !important;
    }}
    div[data-testid="stChatMessageContent"] {{
        width: 100%;
    }}

    .msg-user-wrap {{ display: flex; justify-content: flex-end; }}
    .msg-user {{
        background: {USER_BUBBLE};
        color: {TEXT_PRIMARY};
        padding: 0.65rem 1rem;
        border-radius: 16px;
        font-size: 0.95rem;
        line-height: 1.55;
        max-width: 80%;
    }}

    .msg-assistant {{
        color: {TEXT_PRIMARY};
        font-size: 0.95rem;
        line-height: 1.65;
        padding-right: 0.5rem;
    }}
    .msg-assistant strong {{ color: {TEXT_PRIMARY}; font-weight: 600; }}

    .chart-wrap {{
        background: {BG_ELEVATED};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 0.5rem 0.75rem 0.25rem;
        margin-top: 0.6rem;
    }}
    .chart-hint {{
        font-size: 0.72rem;
        color: {TEXT_SECONDARY};
        text-align: right;
        margin: 0.2rem 0.3rem 0.4rem 0;
    }}

    .empty-state {{
        text-align: center;
        color: {TEXT_SECONDARY};
        padding: 3rem 1rem;
        font-size: 0.88rem;
    }}
    .empty-state .big {{
        font-size: 1.05rem;
        color: {TEXT_PRIMARY};
        font-weight: 600;
        margin-bottom: 0.4rem;
    }}

    /* Override Streamlit's own CSS variables so every native widget
       (chat input, sliders, etc.) follows our toggle, not a fixed base theme */
    :root, .stApp {{
        --background-color: {BG} !important;
        --secondary-background-color: {BG_ELEVATED} !important;
        --text-color: {TEXT_PRIMARY} !important;
    }}

    div[data-testid="stChatInput"] {{
        background: {BG_ELEVATED} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 24px !important;
        max-width: 720px;
        margin: 0 auto;
    }}
    /* Target every nested layer Streamlit may render the textarea inside,
       since the exact wrapper class can vary by version */
    div[data-testid="stChatInput"] * {{
        background: transparent !important;
        color: {TEXT_PRIMARY} !important;
    }}
    div[data-testid="stChatInput"] textarea {{
        background: transparent !important;
        color: {TEXT_PRIMARY} !important;
        caret-color: {TEXT_PRIMARY} !important;
    }}
    div[data-testid="stChatInput"] textarea::placeholder {{
        color: {TEXT_SECONDARY} !important;
        opacity: 1 !important;
    }}

    section[data-testid="stSidebar"] {{
        background-color: {BG_ELEVATED} !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: {TEXT_PRIMARY} !important;
    }}

    header[data-testid="stHeader"] {{ background: transparent; }}
    footer {{ visibility: hidden; }}
    #MainMenu {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="app-header">
    <div class="mark">💳</div>
    <div>
        <div class="name">Spend Insights Assistant</div>
        <div class="desc">Ask anything about your transaction data — answered from real data, running locally</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "collection" not in st.session_state:
    st.session_state.collection = None
if "df" not in st.session_state:
    st.session_state.df = None
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0

# ---------------------------------------------------------------------------
# Data upload
# ---------------------------------------------------------------------------
if st.session_state.collection is None:
    st.markdown("""
    <div class="empty-state">
        <div class="big">📊 Upload your transaction data to begin</div>
        Expected columns: transaction_date, customer_id, merchant_category, amount
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload transactions.csv", type="csv", label_visibility="collapsed")

    if uploaded:
        df = pd.read_csv(uploaded)
        with st.spinner("Indexing your data..."):
            docs = build_summary_documents(df)
            st.session_state.collection = build_vector_store(docs)
            st.session_state.df = df
            st.session_state.doc_count = len(docs)
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"I've indexed **{len(docs)} summaries** from **{len(df)} transactions**. "
                       f"Ask me about spending trends, top categories, or customer segments — "
                       f"I'll show a chart alongside the answer when it helps.",
            "chart": None,
        })
        st.rerun()
else:
    st.markdown(
        f'<div class="status-line"><span class="dot"></span>'
        f'Connected · {st.session_state.doc_count} summaries indexed · running on Ollama (local)</div>',
        unsafe_allow_html=True
    )

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(
                f'<div class="msg-user-wrap"><div class="msg-user">{msg["content"]}</div></div>',
                unsafe_allow_html=True
            )
    else:
        with st.chat_message("assistant"):
            st.markdown(f'<div class="msg-assistant">{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("chart") is not None:
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                st.plotly_chart(themed(msg["chart"]), use_container_width=True,
                                 key=f"chart_{i}", config=PLOTLY_CONFIG)
                st.markdown('<div class="chart-hint">⬇ hover the chart, then click the camera icon to download</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
if st.session_state.collection is not None:
    question = st.chat_input("Ask about your spend data...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question, "chart": None})
        with st.chat_message("user"):
            st.markdown(
                f'<div class="msg-user-wrap"><div class="msg-user">{question}</div></div>',
                unsafe_allow_html=True
            )

        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            answer_placeholder.markdown('<div class="msg-assistant">▍</div>', unsafe_allow_html=True)
            answer = ""
            chart = None
            try:
                for chunk in answer_question_stream(st.session_state.collection, question):
                    answer += chunk
                    answer_placeholder.markdown(f'<div class="msg-assistant">{answer}▍</div>', unsafe_allow_html=True)
                answer_placeholder.markdown(f'<div class="msg-assistant">{answer}</div>', unsafe_allow_html=True)
                chart = maybe_build_chart(question, st.session_state.df)
            except RuntimeError as e:
                answer = f"⚠️ {e}"
                answer_placeholder.markdown(f'<div class="msg-assistant">{answer}</div>', unsafe_allow_html=True)

            if chart is not None:
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                st.plotly_chart(themed(chart), use_container_width=True,
                                 key=f"chart_live_{len(st.session_state.messages)}", config=PLOTLY_CONFIG)
                st.markdown('<div class="chart-hint">⬇ hover the chart, then click the camera icon to download</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": answer, "chart": chart})
