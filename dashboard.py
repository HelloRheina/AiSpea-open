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
from chatbot import initialize_system, ask_parent, proxy
from huggingface_hub import login

proxy()
login(token=os.environ.get("HF_TOKEN", ""))

import logging
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

# import logging
# logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

# ─── 0) emoji dictionary ───────────────────────────────────────────
emotion_with_emoji = {
    "Happy":      "开心 😊",
    "Neutral":    "中性 😐",
    "Surprised":  "惊奇 😲",
    "Questioning":"疑问 🤔",
    "Angry":      "愤怒 😠",
    "Concerned":  "关切 😟",
    "Sad":        "悲伤 😢",
    "Disgusted":  "厌恶 🤢",
}

DATA_PATH      = "data/child_multimodal_dataset.json"
REBUILD_INDEX  = False  # True → force rebuild

@st.cache_resource
def get_system(data_path: str, rebuild: bool):
    # this will only run once per session (or when your code changes)
    return initialize_system(data_path, rebuild)


idx, docs, embedder, pipe = get_system(DATA_PATH, REBUILD_INDEX)

# ───────────────────── word-/char counter ─────────────────────
def word_count(text: str) -> int:
    """
    Return word count for spaced languages, otherwise use jieba to
    segment Chinese and count those tokens.
    """
    if " " in text.strip():
        return len(text.split())
    return len([t for t in jieba.lcut(text) if t.strip()])


# ───────────────────── page style ─────────────────────
st.markdown(
    """
    <style>
    body { background:#FAF3E0; color:#5C3D2E; font-family:Arial, sans-serif; }
    .block-container{ padding:1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────── ──── 1. load dashboard_mock.json ───────────────────
@st.cache_data
def load_dashboard_json() -> dict:
    with open("output/dashboard_mock.json",
              "r", encoding="utf-8") as f:
        return json.load(f)

mock = load_dashboard_json()

# ───────────────────── 2. load child_stats.json ─────────────────────
@st.cache_data
def load_child_stats() -> dict:
    path = "output/child_stats.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

child_stats = load_child_stats()
vocab_size = child_stats.get("vocabulary_size", 0)
avg_utt_len = child_stats.get("average_utterance_length", 0)
# ───────────────────── 3. load predictions.json ─────────────────────
@st.cache_data
def load_predictions() -> pd.DataFrame:
    path = "output/emotion_predictions.json"
    df = pd.read_json(path)

    # synthetic, varied timestamps
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


df = load_predictions()

# ───────────────────── 3. helper functions ─────────────────────
def random_topic_proportions():
    topics = ["Literacy", "Mathematics", "World Knowledge", "Arts & Design"]
    return {t: random.randint(10, 40) for t in topics}


def random_word_frequencies(topic: str):
    vocab = {
        "Literacy": ["reading", "writing", "books", "story", "alphabet"],
        "Mathematics": ["numbers", "counting", "addition", "subtraction", "shapes"],
        "World Knowledge": ["animals", "plants", "weather", "seasons", "space"],
        "Arts & Design": ["painting", "sketching", "music", "dancing", "colors"],
    }
    return {w: random.randint(5, 20) for w in vocab[topic]}


# ───────────────────── 4. sidebar profile & time filter ─────────────────────
st.sidebar.image("dashboard/AiSpea.png", width=300)
st.sidebar.header("👶 儿童档案")
child_name = st.sidebar.text_input("姓名", "猫猫")
child_age = st.sidebar.selectbox("年龄", list(range(1, 11)), index=1)

st.sidebar.header("📅 时间筛选")
preset = st.sidebar.selectbox(
    "预设范围",
    ["全部", "最近7天", "最近30天", "本月", "指定月份", "自定义范围"],
)
today = df["time"].max().date()

if preset == "Specific month":
    months = pd.date_range(df["time"].min().date(), today, freq="MS").strftime("%Y-%m")
    ym = st.sidebar.selectbox("Choose month", months)
    year, mon = map(int, ym.split("-"))
    start_date = date(year, mon, 1)
    end_date = (pd.Timestamp(start_date) + pd.offsets.MonthEnd()).date()
elif preset == "Custom range":
    start_date, end_date = st.sidebar.date_input(
        "Pick date range", [df["time"].min().date(), today]
    )
    if isinstance(start_date, tuple):
        start_date, end_date = start_date[0], start_date[1]
else:
    if preset == "Last 7 days":
        start_date = today - timedelta(days=6)
    elif preset == "Last 30 days":
        start_date = today - timedelta(days=29)
    elif preset == "This month":
        start_date = today.replace(day=1)
    else:
        start_date = df["time"].min().date()
    end_date = today

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
filtered_df = df.loc[(df["time"] >= start_ts) & (df["time"] <= end_ts)].copy()

# recompute cumulative & vocab in-window
filtered_df["Cum Words"] = filtered_df["Word Count"].cumsum()
running_vocab, sizes = set(), []
for sent in filtered_df["sentence"]:
    running_vocab.update(sent.split())
    sizes.append(len(running_vocab))
filtered_df["Running Vocabulary Size"] = sizes
daily_vocab = (
    filtered_df.groupby(filtered_df["time"].dt.date)["Running Vocabulary Size"]
    .max()
    .diff()
    .fillna(0)
    .rename("Daily New Words")
)

# ───────────────────── 5. header & snapshot ─────────────────────
st.title("EchooGrow – 成长仪表盘")
ui.card()
st.markdown(
    (
        f"**显示数据：<span style='color:#EF8985; font-weight:bold;'>{start_date}</span> "
        f"至 <span style='color:#EF8985; font-weight:bold;'>{end_date}</span>  "
        f"(共{len(filtered_df)}次发言)**"
    ),
    unsafe_allow_html=True,
)


ui.card(key="summary")
st.subheader("🐣 概况")

tot_words = int(filtered_df["Cum Words"].iloc[-1]) if not filtered_df.empty else 0
avg_len   = round(filtered_df["Word Count"].mean(), 2) if not filtered_df.empty else 0
main_emotion = (
    filtered_df["emotion"].mode()[0] if not filtered_df.empty else "N/A"
)

st.markdown(
    f"<span style='color:#EF8985; font-weight:bold;'>{child_name}</span>在此期间共使用了"
    f"<span style='color:#EF8985; font-weight:bold;'>{vocab_size}</span>个不同词汇，"
    f"平均每次发言 <span style='color:#EF8985; font-weight:bold;'>{avg_utt_len}</span>个词；"
    f"最常见的情绪为 <span style='color:#EF8985; font-weight:bold;'>{emotion_with_emoji[main_emotion]}</span>。",
    unsafe_allow_html=True,
)

# # ═══════════════════ chatbot  ═══════════════════

# ─── persistent history ─────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def bot_reply(msg: str) -> str:     # ← replace with real LLM
    return textwrap.shorten(
        f"I’m still learning, but {child_name} most often shows "
        f"{main_emotion} and averages {avg_len} words per utterance.",
        width=140, placeholder="…"
    )

# ─── section header ─────────────────────────────────
st.divider()
st.markdown("### 🤖 AI聊天机器人")

# ─── 1) scrolling chat area ─────────────────────────
with st.container(height=250, border=True):
    for role, txt in st.session_state.chat_history:
        clr = "#0068C9" if role == "user" else "#3d8b3d"
        st.markdown(
            f"""<div style="background:{clr};color:#fff;padding:6px 8px;
                        border-radius:6px;margin:6px 0;max-width:95%;
                        font-size:.85rem;">{txt}</div>""",
            unsafe_allow_html=True,
        )

# ─── 2) input row (outside the box, full width) ─────
inp_col, btn_col = st.columns([1.85, 0.2])   # total width = box width
with inp_col:
    user_q = st.text_input(
        "Ask about your child’s progress:",
        key="chat_input",
        label_visibility="collapsed",
        placeholder="您想了解孩子的哪方面内...",
    )


with btn_col:
    if st.button("询问", key="chat_send") and user_q.strip():
        # 1) run your LLM/pipeline only once, here
        ans = ask_parent(user_q, idx, docs, embedder, pipe) 

        # 2) append to history
        st.session_state.chat_history.append(("user", user_q.strip()))
        # st.session_state.chat_history.append(("bot", bot_reply(user_q.strip())))
        st.session_state.chat_history.append(("bot", ans))

        del st.session_state["chat_input"]   # clear box for next run
        st.rerun()



# ───────────────────── 6. linguistic competence bar ─────────────────────
st.subheader("🗣️ 我孩子的语言有多复杂？")
ui.card()

ling_df = pd.json_normalize(mock["linguistic_competence"])

level_info = {
    "L1": {"emoji": "🌱", "desc": "简单词语/短语", "color": "#a6cee3"},
    "L2": {"emoji": "🌿", "desc": "基础句子",     "color": "#1f78b4"},
    "L3": {"emoji": "🌳", "desc": "复杂表达",     "color": "#08519c"},
}

ling_df["Label"]  = ling_df["level"].apply(
    lambda x: f"{x} {level_info[x]['emoji']}"
)
ling_df["Color"]  = ling_df["level"].apply(lambda x: level_info[x]["color"])

fig_ling = go.Figure(
    go.Bar(
        x=ling_df["Label"],
        y=ling_df["count"],
        marker_color=ling_df["Color"],
        width=0.6,
        customdata=ling_df[["example"]],
        hovertemplate="<b>%{x}</b><br>数量: %{y}<br>示例: %{customdata[0]}<extra></extra>",
    )
)
# fig_ling.update_layout(
#     title="语言能力分布",
#     yaxis_title="出现次数",
#     showlegend=False,
#     height=400,
# )
st.plotly_chart(fig_ling, use_container_width=True)


# ───────── 7a.Emotional-Flow Complexity – E1 · E2 · E3  (Pie Chart) ──────
st.subheader("🎭 对话中的情感变化有哪些？")
ui.card()
# 1) raw counts  ----------------------------------------------------------
e_flow = {"E1": 45, "E2": 30, "E3": 25}
e_df   = (
    pd.Series(e_flow, name="数量")
      .rename_axis("E-level")
      .reset_index()
)

# add example text for hover
example_map = {
    "E1": "示例：持续好奇",
    "E2": "示例：生气→困惑→释然",
    "E3": "示例：焦虑→共情→创造性解决",
}
e_df["示例"] = e_df["E-level"].map(example_map)

# ─── pie chart ───────────────────────────────────────────────────
fig_e_pie = px.pie(
    e_df,
    names="E-level",                # labels are plain E1 / E2 / E3
    values="数量",
    color="E-level",
    color_discrete_map={"E1": "#66c2a5", "E2": "#fc8d62", "E3": "#8da0cb"},
    hole=0.35,
    labels={"E-level": "持续好奇", "数量": "数量"},   # legend title stays 中文
)

fig_e_pie.update_traces(
    textposition="inside",
    texttemplate="%{label}<br>%{percent:.0%}",
    customdata=e_df["示例"],
    hovertemplate="数量: %{percent:.0%}<br>%{customdata}<extra></extra>"   # one-line count→percent
)

fig_e_pie.update_layout(
    width=550,
    height=550,
    margin=dict(l=20, r=20, t=40, b=20),
    legend_title_text="点击筛选",
)

st.plotly_chart(fig_e_pie, use_container_width=False)

# ───────────────────── 7b. emotion bar (chunky) ─────────────────────

# ─── 1) counts and palette ─────────────────────────────────────────
emo_counts = (
    filtered_df["emotion"]
    .value_counts()
    .rename_axis("emotion")
    .reset_index(name="count")
)

# map plain → emoji label; fall back to original text if not found
emo_counts["label"] = emo_counts["emotion"].map(emotion_with_emoji) \
                                           .fillna(emo_counts["emotion"])

palette   = px.colors.qualitative.Pastel
col_map   = {e: palette[i % len(palette)]
             for i, e in enumerate(emo_counts["emotion"])}

# ─── 2) bar chart with emoji labels ────────────────────────────────
ui.card()
fig_e = go.Figure(
    go.Bar(
        x=emo_counts["label"],                         # ← emoji labels
        y=emo_counts["count"],
        marker_color=[col_map[e] for e in emo_counts["emotion"]],
        width=[0.85] * len(emo_counts),
    )
)
fig_e.update_layout(
    title="情绪分布",
    bargap=0.02,
    height=400,
    yaxis_title="数量",
)

st.plotly_chart(fig_e, use_container_width=True)


# ───────────────────── 8. Interest Orientation – Sunburst ─────────────────────
# ───────────────────── Interest Orientation – Sunburst (purple, blue, yellow) ─────────────────────
st.subheader("🌞 我孩子对哪些话题感兴趣？")
ui.card()

# build tidy DF → Domain / Subtopic / Count
rows = [
    (dom, sub, cnt)
    for dom, subs in mock["interest_orientation"].items()
    for sub, cnt in subs.items()
]
sun_df = pd.DataFrame(rows, columns=["Domain", "Subtopic", "Count"])

# purple, green, orange palette
domain_colors = {
    "STEM探索": "#9B59B6",   # purple
    "创意表达": "#2ECC71",   # green
    "社会情感": "#E67E22",   # orange
}

fig_sun = px.sunburst(
    sun_df,
    path=["Domain", "Subtopic"],
    values="Count",
    color="Domain",
    color_discrete_map=domain_colors,
)

# ➊ show % on the chart  ➋ add % in hover tooltip
fig_sun.update_traces(
    textinfo="label+percent entry",                  # label + % of total
    hovertemplate=(
        "%{label}<br>"
        "对话量: %{value}<br>"
        "占比: %{percentEntry:.1%}"
        "<extra></extra>"
    ),
)

fig_sun.update_layout(
    width=500,
    height=500,
    margin=dict(t=40, l=10, r=10, b=10),
)

st.plotly_chart(fig_sun, use_container_width=False)



# ───────────────────── 9. Word-Cloud Explorer ─────────────────────
st.subheader("☁️ 词云探索器")

# Load the real data
child_stats = load_child_stats()

# Extract word frequencies from the data
vocab_freq = {item['token']: item['count'] for item in child_stats['vocabulary_frequency']}

# Try common Chinese font paths - use the first one that works
chinese_font_paths = [
    "dashboard/NotoSansSC-VariableFont_wght.ttf",
    None  # Fallback to default
]

# Initialize word cloud
wc = None

for font_path in chinese_font_paths:
    try:
        # Create word cloud with parameters optimized for Chinese
        wc = WordCloud(
            font_path=font_path,
            width=800,
            height=400,
            background_color="white",
            max_words=200,
            max_font_size=120,
            min_font_size=10,
            prefer_horizontal=0.9,  # Better for Chinese characters
            colormap='viridis',
            scale=2
        )
        wc.generate_from_frequencies(vocab_freq)
        break  # Exit loop if successful
    except Exception as e:
        st.warning(f"Attempt with font {font_path} failed: {str(e)}")
        continue

if wc is None:
    try:
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white"
        )
        wc.generate_from_frequencies(vocab_freq)
    except Exception as e:
        st.error(f"Failed to generate word cloud: {str(e)}")
        st.stop()

# Display the word cloud directly without PNG conversion
if wc:
    try:
        # Get the PIL Image directly
        wc_image = wc.to_image()
        
        # Display using Streamlit's native image display
        st.image(wc_image)
        
        # Download button
        img_bytes = io.BytesIO()
        wc_image.save(img_bytes, format='PNG')
        st.download_button(
            label="下载词云",
            data=img_bytes.getvalue(),
            file_name="wordcloud.png",
            mime="image/png"
        )
    except Exception as e:
        st.error(f"Error handling word cloud image: {str(e)}")


# ───────────────────── 10. parent observations ─────────────────────
st.subheader("👀 家长观察记录")

# 1. Initialize session state
if 'observations' not in st.session_state:
    st.session_state.observations = []

# 2. Add new observation
with st.expander("➕ 添加新记录", expanded=False):
    observation = st.text_area(
        "写下您今天的观察：",
        placeholder="例：今天孩子第一次完整唱完一首儿歌！",
        key="new_obs"
    )
    
    if st.button("💾 保存记录"):
        if observation.strip():
            new_record = {
                "date": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
                "text": observation.strip(),
                "child": child_name  # Using the child_name from your sidebar
            }
            st.session_state.observations.append(new_record)
            st.success("记录已保存！")
            st.rerun()
        else:
            st.warning("请输入观察内容")

st.write("")
st.write("")

# 3. Display all observations
st.subheader("📚 历史观察记录")

if not st.session_state.observations:
    st.info("暂无记录，请添加第一条观察记录")
else:
    # Create filter controls
    cols = st.columns(2)
    
    with cols[0]:
        # Get unique months
        months = sorted(list(set(obs['date'][:7] for obs in st.session_state.observations)))
        filter_month = st.selectbox("筛选月份", ["所有月份"] + months)
    
    with cols[1]:
        sort_order = st.radio("排序方式", ["从新到旧", "从旧到新"], horizontal=True)

    # Filter and sort
    filtered_obs = [
        obs for obs in st.session_state.observations
        if filter_month == "所有月份" or obs['date'].startswith(filter_month)
    ]
    
    filtered_obs.sort(key=lambda x: x['date'], reverse=(sort_order == "从新到旧"))

    # Display observations
    for i, obs in enumerate(filtered_obs, 1):
        with st.expander(f"📅 {obs['date']}", expanded=False):
            st.markdown(f"""
            **👶 孩子:** {obs.get('child', child_name)}  
            **📝 内容:** {obs['text']}
            """)
            
            if st.button(f"🗑️ 删除", key=f"del_{i}"):
                st.session_state.observations.remove(obs)
                st.rerun()

# ───────────────────── 11. raw data ─────────────────────
# with st.expander("📄 Raw data"):
#     st.dataframe(filtered_df)