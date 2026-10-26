# AiSpea: AI-Powered Child Development Analysis for Smart Toys

AiSpea analyzes child–toy conversation logs to generate developmental reports for parents, covering **language complexity**, **emotional patterns**, and **interest orientation**. Built for the 2025 Global Artificial Intelligence Innovation Competition, where it reached the **national top 16 (Second Prize)** as a 5-person team project.

## Features

### 1. RAG-based Parent Chatbot
Parents ask questions about their child's development in natural language. The system retrieves relevant conversation snippets via FAISS similarity search and generates personalized answers using **Gemma-3-4B**.

### 2. Emotion Detection
Classifies each child utterance into 8 emotion categories (Happy, Neutral, Surprised, Questioning, Angry, Concerned, Sad, Disgusted) using a fine-tuned Chinese emotion model ([Johnson8187/Chinese-Emotion](https://huggingface.co/Johnson8187/Chinese-Emotion)).

### 3. Linguistic Competence Assessment
Classifies utterances into three complexity levels using DistilBERT with LoRA fine-tuning:
- **L1**: simple words/phrases (e.g., "积木倒了")
- **L2**: basic sentences with reasoning (e.g., "叶子掉下来是因为风大")
- **L3**: complex/creative expression (e.g., "如果声音跑得慢，雷声就像迟到的快递员")

### 4. Interest Orientation
Identifies the child's interests across three domains (STEM, Creative Expression, Social-Emotional) using both keyword matching and LLM-based classification (Qwen 1.5-1.8B).

### 5. Interactive Dashboard
A Streamlit dashboard presenting all analyses with:
- Emotion distribution bar charts and emotional-flow pie charts
- Linguistic complexity breakdown
- Interest orientation sunburst diagram
- Word cloud from the child's vocabulary
- Time-range filtering and parent observation journal
- Integrated RAG chatbot for follow-up questions

## Architecture

```
Child–Toy Conversation Logs (JSON)
         │
         ├──► Emotion Detection ──────────► emotion_predictions.json
         │    (Chinese-Emotion model)
         │
         ├──► Linguistic Competence ──────► dashboard_mock.json (L1/L2/L3)
         │    (DistilBERT + LoRA)
         │
         ├──► Interest Orientation ──────► interest_predictions.json
         │    (keyword matching + Qwen)
         │
         ├──► FAISS Index ───────────────► RAG chatbot (Gemma-3-4B)
         │    (text2vec-large-chinese)
         │
         └──► Dashboard (Streamlit) ─────► Parent-facing UI
```

## Repository Structure

```
AiSpea/
├── dashboard.py               # Streamlit dashboard + integrated RAG chatbot
├── chatbot.py                 # RAG pipeline: FAISS retrieval + Gemma-3 generation
├── dataset.py                 # PyTorch Dataset for child dialogue JSON
├── emotional_flow.py          # Emotion classification (Chinese-Emotion model)
├── interest_orientation.py    # Keyword-based interest detection
├── interest_LLM.py            # LLM-based interest classification (Qwen)
├── linguistic_competence.ipynb # DistilBERT + LoRA fine-tuning notebook
├── translate-goemo.py         # GoEmotions EN→ZH translation via DeepSeek API
├── data/
│   ├── child_multimodal_dataset.json  # 200 annotated child–AI conversations
│   ├── dataset.json                    # Earlier dataset version
│   └── goemotions_train_1k_zh.json    # Translated GoEmotions subset
├── output/                    # Pre-computed analysis results
│   ├── child_stats.json       # Vocabulary stats
│   ├── emotion_predictions.json
│   ├── interest_predictions.json
│   └── dashboard_mock.json    # Aggregated dashboard data
├── dashboard/                 # Dashboard assets
│   ├── AiSpea.png             # Logo
│   ├── AiSpea_black.png       # Logo (dark variant)
│   └── NotoSansSC-VariableFont_wght.ttf  # Chinese font for word cloud
├── requirements.txt
└── .gitignore
```

## Installation

**Prerequisites:** CUDA-capable GPU (for LLM inference), Python 3.11+.

```bash
git clone https://github.com/Hyrsta/AiSpea.git
cd AiSpea

conda create -n aispea python=3.11 -y
conda activate aispea
```

### Install dependencies

```bash
# PyTorch (match your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Core ML libraries
pip install -r requirements.txt

# Dashboard UI
pip install streamlit streamlit-shadcn-ui plotly wordcloud jieba

# RAG pipeline
pip install faiss-cpu sentence-transformers romatch
```

### Environment setup

```bash
# HuggingFace token (required for Gemma-3 access)
export HF_TOKEN=your_huggingface_token

# Optional: DeepSeek API key (only needed for translate-goemo.py)
# Create a .env file with: DEEPSEEK_KEY=your_key
```

## Usage

### Run the dashboard (full system)

```bash
streamlit run dashboard.py
```

This starts the Streamlit dashboard with the integrated RAG chatbot. On first run, it builds a FAISS index from the conversation data and loads the Gemma-3 model.

### Run individual modules

```bash
# Emotion detection
python emotional_flow.py

# Interest analysis (keyword-based)
python interest_orientation.py

# Interest analysis (LLM-based)
python interest_LLM.py

# RAG chatbot (CLI mode)
python chatbot.py
```

## Data Format

Input conversations follow this JSON structure:

```json
[
  {
    "metadata": {
      "年龄": "8岁",
      "对话场景": ["幻想游戏"],
      "语言类型": "口语化非正式表达"
    },
    "dialog": [
      {"role": "child", "text": "我们今天玩个魔法师冒险的游戏吧！"},
      {"role": "AI", "text": "好啊！你想当魔法师还是探险者？"},
      {"role": "child", "text": "当然是最厉害的魔法师了！我还能召唤火龙。"}
    ],
    "labels": {
      "语言能力": "L3",
      "情感变化": "E1",
      "兴趣倾向": ["社会情感/自我认知"]
    }
  }
]
```

## Models Used

| Component | Model | Source |
|-----------|-------|--------|
| RAG embeddings | text2vec-large-chinese | [GanymedeNil/text2vec-large-chinese](https://huggingface.co/GanymedeNil/text2vec-large-chinese) |
| RAG generation | Gemma-3-4B-IT | [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) |
| Emotion classification | Chinese-Emotion | [Johnson8187/Chinese-Emotion](https://huggingface.co/Johnson8187/Chinese-Emotion) |
| Linguistic competence | DistilBERT-zh + LoRA | [Geotrend/distilbert-base-zh-cased](https://huggingface.co/Geotrend/distilbert-base-zh-cased) |
| Interest classification | Qwen 1.5-1.8B | [Qwen/Qwen1.5-1.8B-Chat](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat) |
| Translation | DeepSeek Chat | [DeepSeek API](https://api.deepseek.com) |

## Team

Built by a 5-person team at Northwestern Polytechnical University for the 2025 Global Artificial Intelligence Innovation Competition.

## License

This project is for educational and research purposes.
