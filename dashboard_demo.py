"""
AiSpea Dashboard — Demo Mode (no GPU required)
Displays all charts from pre-computed output/ data.
Chatbot section is stubbed out.
Supports dark mode toggle + English/Chinese localization.
"""
import json
import os
import random
import io
from datetime import date, datetime, timedelta

import jieba
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit_shadcn_ui as ui
from wordcloud import WordCloud
import textwrap

import logging
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

# ═══════════════════════════════════════════════════════════════════
# LOCALIZATION
# ═══════════════════════════════════════════════════════════════════
STRINGS = {
    "zh": {
        "title": "EchooGrow – 成长仪表盘",
        "child_profile": "👶 儿童档案",
        "name": "姓名",
        "default_name": "猫猫",
        "age": "年龄",
        "time_filter": "📅 时间筛选",
        "preset": "预设范围",
        "presets": ["全部", "最近7天", "最近30天", "本月", "自定义范围"],
        "pick_range": "选择日期范围",
        "data_range": "显示数据：**{start}** 至 **{end}** (共{n}次发言)",
        "overview": "🐣 概况",
        "overview_text": (
            "<span style='color:{accent}; font-weight:bold;'>{name}</span>在此期间共使用了"
            "<span style='color:{accent}; font-weight:bold;'>{vocab}</span>个不同词汇，"
            "平均每次发言 <span style='color:{accent}; font-weight:bold;'>{avg_len}</span>个词；"
            "最常见的情绪为 <span style='color:{accent}; font-weight:bold;'>{emotion}</span>。"
        ),
        "chatbot": "### 🤖 AI聊天机器人",
        "chatbot_stub": "💡 演示模式：聊天机器人需要 GPU 运行 Gemma-3-4B 模型。",
        "linguistic": "🗣️ 我孩子的语言有多复杂？",
        "ling_levels": {"L1": "简单词语/短语", "L2": "基础句子", "L3": "复杂表达"},
        "ling_hover": "<b>%{x}</b><br>数量: %{y}<br>示例: %{customdata[0]}<extra></extra>",
        "emotional_flow": "🎭 对话中的情感变化有哪些？",
        "e_examples": {"E1": "示例：持续好奇", "E2": "示例：生气→困惑→释然", "E3": "示例：焦虑→共情→创造性解决"},
        "emotion_dist": "情绪分布",
        "count": "数量",
        "interest": "🌞 我孩子对哪些话题感兴趣？",
        "interest_hover": "%{label}<br>对话量: %{value}<br>占比: %{percentEntry:.1%}<extra></extra>",
        "click_filter": "点击筛选",
        "wordcloud": "☁️ 词云探索器",
        "observations": "👀 家长观察记录",
        "add_record": "➕ 添加新记录",
        "obs_placeholder": "例：今天孩子第一次完整唱完一首儿歌！",
        "obs_prompt": "写下您今天的观察：",
        "save": "💾 保存记录",
        "saved": "记录已保存！",
        "no_records": "暂无记录，请添加第一条观察记录",
        "history": "📚 历史观察记录",
        "child_label": "孩子",
        "content_label": "内容",
        "delete": "🗑️ 删除",
        "settings": "⚙️ 设置",
        "dark_mode": "🌙 深色模式",
        "language": "🌐 语言",
        "download_wc": "下载词云",
        "emotion_emoji": {
            "Happy": "开心 😊", "Neutral": "中性 😐", "Surprised": "惊讶 😲",
            "Questioning": "好奇 🤔", "Angry": "生气 😠", "Concerned": "担忧 😟",
            "Sad": "伤心 😢", "Disgusted": "厌恶 🤢",
        },
    },
    "en": {
        "title": "EchooGrow – Child Development Dashboard",
        "child_profile": "👶 Child Profile",
        "name": "Name",
        "default_name": "Mao Mao",
        "age": "Age",
        "time_filter": "📅 Time Filter",
        "preset": "Preset Range",
        "presets": ["All", "Last 7 Days", "Last 30 Days", "This Month", "Custom Range"],
        "pick_range": "Pick date range",
        "data_range": "Showing data: **{start}** to **{end}** ({n} utterances)",
        "overview": "🐣 Overview",
        "overview_text": (
            "<span style='color:{accent}; font-weight:bold;'>{name}</span> used "
            "<span style='color:{accent}; font-weight:bold;'>{vocab}</span> unique words, "
            "averaging <span style='color:{accent}; font-weight:bold;'>{avg_len}</span> words per utterance. "
            "Most common emotion: <span style='color:{accent}; font-weight:bold;'>{emotion}</span>."
        ),
        "chatbot": "### 🤖 AI Chatbot",
        "chatbot_stub": "💡 Demo mode: the chatbot requires a GPU to run Gemma-3-4B.",
        "linguistic": "🗣️ How Complex Is My Child's Language?",
        "ling_levels": {"L1": "Simple words/phrases", "L2": "Basic sentences", "L3": "Complex expression"},
        "ling_hover": "<b>%{x}</b><br>Count: %{y}<br>Example: %{customdata[0]}<extra></extra>",
        "emotional_flow": "🎭 What Emotional Shifts Occur in Conversations?",
        "e_examples": {"E1": "e.g. sustained curiosity", "E2": "e.g. angry → confused → relieved", "E3": "e.g. anxious → empathetic → creative solution"},
        "emotion_dist": "Emotion Distribution",
        "count": "Count",
        "interest": "🌞 What Topics Interest My Child?",
        "interest_hover": "%{label}<br>Conversations: %{value}<br>Share: %{percentEntry:.1%}<extra></extra>",
        "click_filter": "Click to filter",
        "wordcloud": "☁️ Word Cloud Explorer",
        "observations": "👀 Parent Observations",
        "add_record": "➕ Add New Record",
        "obs_placeholder": "e.g. Today my child sang a full song for the first time!",
        "obs_prompt": "Write down your observation today:",
        "save": "💾 Save",
        "saved": "Record saved!",
        "no_records": "No records yet — add your first observation",
        "history": "📚 Observation History",
        "child_label": "Child",
        "content_label": "Content",
        "delete": "🗑️ Delete",
        "settings": "⚙️ Settings",
        "dark_mode": "🌙 Dark Mode",
        "language": "🌐 Language",
        "download_wc": "Download Word Cloud",
        "emotion_emoji": {
            "Happy": "Happy 😊", "Neutral": "Neutral 😐", "Surprised": "Surprised 😲",
            "Questioning": "Curious 🤔", "Angry": "Angry 😠", "Concerned": "Concerned 😟",
            "Sad": "Sad 😢", "Disgusted": "Disgusted 🤢",
        },
    },
}

# Interest domain translations
DOMAIN_NAMES = {
    "zh": {"STEM探索": "STEM探索", "创意表达": "创意表达", "社会情感": "社会情感"},
    "en": {"STEM探索": "STEM Exploration", "创意表达": "Creative Expression", "社会情感": "Social-Emotional"},
}

# ═══════════════════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ═══════════════════════════════════════════════════════════════════
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "lang" not in st.session_state:
    st.session_state.lang = "zh"
if "observations" not in st.session_state:
    st.session_state.observations = []

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR — Settings
# ═══════════════════════════════════════════════════════════════════
lang = st.session_state.lang
t = STRINGS[lang]

st.sidebar.image(
    "dashboard/AiSpea_black.png" if st.session_state.dark_mode else "dashboard/AiSpea.png",
    width=300,
)

st.sidebar.markdown(f"#### {t['settings']}")
dark = st.sidebar.toggle(t["dark_mode"], value=st.session_state.dark_mode, key="dark_toggle")
if dark != st.session_state.dark_mode:
    st.session_state.dark_mode = dark
    st.rerun()

lang_choice = st.sidebar.radio(t["language"], ["中文", "English"], index=0 if lang == "zh" else 1, horizontal=True)
new_lang = "zh" if lang_choice == "中文" else "en"
if new_lang != st.session_state.lang:
    st.session_state.lang = new_lang
    st.rerun()

# Re-read after potential rerun
lang = st.session_state.lang
t = STRINGS[lang]

# ═══════════════════════════════════════════════════════════════════
# THEME CSS
# ═══════════════════════════════════════════════════════════════════
if st.session_state.dark_mode:
    accent = "#FF6B6B"
    bg = "#1a1a2e"
    card_bg = "#16213e"
    text_color = "#e0e0e0"
    chart_template = "plotly_dark"
    wc_bg = "#1a1a2e"
    st.markdown(f"""
    <style>
    .stApp {{ background-color: {bg}; color: {text_color}; }}
    .stSidebar > div:first-child {{ background-color: {card_bg}; }}
    .stMarkdown, .stText {{ color: {text_color}; }}
    h1, h2, h3, h4, h5, h6 {{ color: #ffffff !important; }}
    .block-container {{ padding: 1rem; }}
    </style>
    """, unsafe_allow_html=True)
else:
    accent = "#EF8985"
    bg = "#FAF3E0"
    card_bg = "#ffffff"
    text_color = "#5C3D2E"
    chart_template = "plotly_white"
    wc_bg = "white"
    st.markdown(f"""
    <style>
    body {{ background: {bg}; color: {text_color}; font-family: Arial, sans-serif; }}
    .block-container {{ padding: 1rem; }}
    </style>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# HELPER — word count
# ═══════════════════════════════════════════════════════════════════
def word_count(text: str) -> int:
    if " " in text.strip():
        return len(text.split())
    return len([tok for tok in jieba.lcut(text) if tok.strip()])

# ═══════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════
@st.cache_data
def load_dashboard_json() -> dict:
    with open("output/dashboard_mock.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_child_stats() -> dict:
    with open("output/child_stats.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_predictions() -> pd.DataFrame:
    df = pd.read_json("output/emotion_predictions.json")
    random.seed(42)  # deterministic timestamps
    start = datetime(2025, 5, 2, 13, 0)
    gaps = [
        timedelta(minutes=random.randint(2, 90))
        if random.random() < 0.8
        else timedelta(hours=random.randint(2, 6))
        if random.random() < 0.9
        else timedelta(days=1)
        for _ in range(len(df) - 1)
    ]
    times = [start]
    for g in gaps:
        times.append(times[-1] + g)
    df["time"] = times[::-1]
    df["sentence"] = df["text"]
    df["sentiment_score"] = df["score"]
    df["Word Count"] = df["sentence"].apply(word_count)
    running_vocab, sizes = set(), []
    for sent in df["sentence"]:
        running_vocab.update(sent.split())
        sizes.append(len(running_vocab))
    df["Running Vocabulary Size"] = sizes
    df["Total Word Count"] = df["Word Count"].cumsum()
    return df

mock = load_dashboard_json()
child_stats = load_child_stats()
vocab_size = child_stats.get("vocabulary_size", 0)
avg_utt_len = child_stats.get("average_utterance_length", 0)
df = load_predictions()

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR — Child profile & time filter
# ═══════════════════════════════════════════════════════════════════
st.sidebar.markdown("---")
st.sidebar.header(t["child_profile"])
child_name = st.sidebar.text_input(t["name"], t["default_name"])
child_age = st.sidebar.selectbox(t["age"], list(range(1, 11)), index=1)

st.sidebar.header(t["time_filter"])
preset_labels = t["presets"]
preset = st.sidebar.selectbox(t["preset"], preset_labels)
today = df["time"].max().date()

if preset == preset_labels[4]:  # Custom range
    start_date, end_date = st.sidebar.date_input(t["pick_range"], [df["time"].min().date(), today])
    if isinstance(start_date, tuple):
        start_date, end_date = start_date[0], start_date[1]
else:
    if preset == preset_labels[1]:  # Last 7 days
        start_date = today - timedelta(days=6)
    elif preset == preset_labels[2]:  # Last 30 days
        start_date = today - timedelta(days=29)
    elif preset == preset_labels[3]:  # This month
        start_date = today.replace(day=1)
    else:
        start_date = df["time"].min().date()
    end_date = today

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
filtered_df = df.loc[(df["time"] >= start_ts) & (df["time"] <= end_ts)].copy()
filtered_df["Cum Words"] = filtered_df["Word Count"].cumsum()

emotion_emoji = t["emotion_emoji"]

# ═══════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════

# ─── Title & Date Range ────────────────────────────────────────
st.title(t["title"])
st.markdown(
    t["data_range"].format(start=start_date, end=end_date, n=len(filtered_df)),
    unsafe_allow_html=True,
)

# ─── Overview ──────────────────────────────────────────────────
st.subheader(t["overview"])
main_emotion = filtered_df["emotion"].mode()[0] if not filtered_df.empty else "N/A"
avg_len = round(filtered_df["Word Count"].mean(), 2) if not filtered_df.empty else 0
st.markdown(
    t["overview_text"].format(
        accent=accent,
        name=child_name,
        vocab=vocab_size,
        avg_len=avg_utt_len,
        emotion=emotion_emoji.get(main_emotion, main_emotion),
    ),
    unsafe_allow_html=True,
)

# ─── Chatbot Stub ─────────────────────────────────────────────
st.divider()
st.markdown(t["chatbot"])
st.info(t["chatbot_stub"])

# ─── Linguistic Competence ─────────────────────────────────────
st.subheader(t["linguistic"])
ling_df = pd.json_normalize(mock["linguistic_competence"])
level_info = {
    "L1": {"emoji": "🌱", "color": "#a6cee3"},
    "L2": {"emoji": "🌿", "color": "#1f78b4"},
    "L3": {"emoji": "🌳", "color": "#08519c"},
}
ling_df["Label"] = ling_df["level"].apply(
    lambda x: f"{x} {level_info[x]['emoji']} {t['ling_levels'][x]}"
)
ling_df["Color"] = ling_df["level"].apply(lambda x: level_info[x]["color"])
fig_ling = go.Figure(go.Bar(
    x=ling_df["Label"], y=ling_df["count"],
    marker_color=ling_df["Color"], width=0.6,
    customdata=ling_df[["example"]],
    hovertemplate=t["ling_hover"],
))
fig_ling.update_layout(template=chart_template, yaxis_title=t["count"])
st.plotly_chart(fig_ling, use_container_width=True)

# ─── Emotional Flow Pie ───────────────────────────────────────
st.subheader(t["emotional_flow"])
e_flow = {"E1": 45, "E2": 30, "E3": 25}
e_df = pd.Series(e_flow, name=t["count"]).rename_axis("E-level").reset_index()
e_df["example"] = e_df["E-level"].map(t["e_examples"])
fig_e_pie = px.pie(
    e_df, names="E-level", values=t["count"], color="E-level",
    color_discrete_map={"E1": "#66c2a5", "E2": "#fc8d62", "E3": "#8da0cb"}, hole=0.35,
)
fig_e_pie.update_traces(
    textposition="inside", texttemplate="%{label}<br>%{percent:.0%}",
    customdata=e_df["example"],
    hovertemplate=f"{t['count']}: %{{percent:.0%}}<br>%{{customdata}}<extra></extra>",
)
fig_e_pie.update_layout(
    template=chart_template, width=550, height=550,
    margin=dict(l=20, r=20, t=40, b=20),
    legend_title_text=t["click_filter"],
)
st.plotly_chart(fig_e_pie, use_container_width=False)

# ─── Emotion Bar ──────────────────────────────────────────────
emo_counts = filtered_df["emotion"].value_counts().rename_axis("emotion").reset_index(name="count")
emo_counts["label"] = emo_counts["emotion"].map(emotion_emoji).fillna(emo_counts["emotion"])
palette = px.colors.qualitative.Pastel
col_map = {e: palette[i % len(palette)] for i, e in enumerate(emo_counts["emotion"])}
fig_e = go.Figure(go.Bar(
    x=emo_counts["label"], y=emo_counts["count"],
    marker_color=[col_map[e] for e in emo_counts["emotion"]],
    width=[0.85] * len(emo_counts),
))
fig_e.update_layout(
    template=chart_template,
    title=t["emotion_dist"], bargap=0.02, height=400, yaxis_title=t["count"],
)
st.plotly_chart(fig_e, use_container_width=True)

# ─── Interest Sunburst ────────────────────────────────────────
st.subheader(t["interest"])
domain_map = DOMAIN_NAMES[lang]
rows = [
    (domain_map.get(dom, dom), sub, cnt)
    for dom, subs in mock["interest_orientation"].items()
    for sub, cnt in subs.items()
]
sun_df = pd.DataFrame(rows, columns=["Domain", "Subtopic", "Count"])
domain_colors_mapped = {domain_map.get(k, k): v for k, v in
    {"STEM探索": "#9B59B6", "创意表达": "#2ECC71", "社会情感": "#E67E22"}.items()}
fig_sun = px.sunburst(
    sun_df, path=["Domain", "Subtopic"], values="Count",
    color="Domain", color_discrete_map=domain_colors_mapped,
)
fig_sun.update_traces(
    textinfo="label+percent entry",
    hovertemplate=t["interest_hover"],
)
fig_sun.update_layout(
    template=chart_template, width=500, height=500,
    margin=dict(t=40, l=10, r=10, b=10),
)
st.plotly_chart(fig_sun, use_container_width=False)

# ─── Word Cloud ───────────────────────────────────────────────
st.subheader(t["wordcloud"])
vocab_freq = {item["token"]: item["count"] for item in child_stats["vocabulary_frequency"]}
try:
    wc = WordCloud(
        font_path="dashboard/NotoSansSC-VariableFont_wght.ttf",
        width=800, height=400, background_color=wc_bg, max_words=200,
        max_font_size=120, min_font_size=10, prefer_horizontal=0.9,
        colormap="cool" if st.session_state.dark_mode else "viridis", scale=2,
    )
    wc.generate_from_frequencies(vocab_freq)
    wc_image = wc.to_image()
    st.image(wc_image)
    img_bytes = io.BytesIO()
    wc_image.save(img_bytes, format="PNG")
    st.download_button(t["download_wc"], data=img_bytes.getvalue(), file_name="wordcloud.png", mime="image/png")
except Exception as e:
    st.error(f"Word cloud error: {e}")

# ─── Parent Observations ──────────────────────────────────────
st.subheader(t["observations"])
with st.expander(t["add_record"], expanded=False):
    observation = st.text_area(t["obs_prompt"], placeholder=t["obs_placeholder"], key="new_obs")
    if st.button(t["save"]):
        if observation.strip():
            st.session_state.observations.append({
                "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                "text": observation.strip(),
                "child": child_name,
            })
            st.success(t["saved"])
            st.rerun()

st.subheader(t["history"])
if not st.session_state.observations:
    st.info(t["no_records"])
else:
    for i, obs in enumerate(st.session_state.observations, 1):
        with st.expander(f"📅 {obs['date']}", expanded=False):
            st.markdown(
                f"**👶 {t['child_label']}:** {obs.get('child', child_name)}  \n"
                f"**📝 {t['content_label']}:** {obs['text']}"
            )
            if st.button(t["delete"], key=f"del_{i}"):
                st.session_state.observations.remove(obs)
                st.rerun()
