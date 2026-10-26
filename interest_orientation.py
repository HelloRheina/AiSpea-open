import re
from typing import Dict, List
from dataset import load_dialogue_data
from torch.utils.data import DataLoader
import json

# Core Domains & Sub-Interests
INTEREST_LABEL = {
    "STEM探索": {
        "物理现象",
        "生物观察"
    },
    "创意表达": {
        "艺术创作",
        "角色扮演"
    },
    "社会情感": {
        "冲突解决",
        "自我认知"
    }
}

INTEREST_MAP = {
    "STEM( 科学、技术、工程与,数学)": {
        "通用科学": ["动物", "植物", "实验", "观察", "自然", "生物学"],
        "天文学": ["星星", "月亮", "行星", "太空", "望远镜", "银河"],
        "数学": ["数学", "数字", "计数", "计算", "加法", "方程"],
        "工程学": ["建造", "设计", "桥梁", "结构", "乐高", "构造"],
        "机器人学": ["机器人", "传感器", "机器", "人工智能", "编码", "编程"]
    },
    "语言与识字": {
        "讲故事": ["故事", "讲", "叙述", "角色", "情节"],
        "阅读": ["书籍", "阅读", "图书馆", "小说"],
        "写作": ["写", "句子", "诗", "文章"]
    },
    "艺术": {
        "视觉艺术": ["画", "绘画", "素描", "上色", "蜡笔", "图片", "独角兽"],
        "音乐": ["歌曲", "音乐", "唱歌", "乐器", "旋律"],
        "表演艺术": ["舞蹈", "戏剧", "表演", "舞台"]
    },
    "社会研究": {
        "地理": ["地图", "国家", "旅行", "大陆"],
        "历史": ["过去", "古代", "战争", "时间线"],
        "公民教育": ["社区", "领导", "规则", "投票"]
    },
}
def collate_text(batch):
    return [item["dialog_item"]["text"] for item in batch if item["dialog_item"]["role"] in "child"]

def identify_interest_loader(loader: DataLoader) -> List[Dict[str, str]]:
    """
    Returns a list of all (domain, sub_interest, score) entries
    sorted descending by score, where score = number of keyword matches
    across the entire loader.
    """
    # 1) Pull every utterance into one big string
    all_texts = []
    for batch in loader:
        all_texts.extend(batch)
    combined = " ".join(all_texts).lower()

    # 2) Count matches per (domain, subcat)
    scores: Dict[tuple, int] = {}
    for domain, subcats in INTEREST_MAP.items():
        for subcat, keywords in subcats.items():
            # simple substring match (good for Chinese too)
            count = sum(1 for kw in keywords if kw in combined)
            if count > 0:
                scores[(domain, subcat)] = count

    # 3) Sort descending by count
    sorted_items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    # 4) Build output list
    return [
        {
            "core_interest": domain,
            "sub_interest": subcat,
            "score": count
        }
        for ((domain, subcat), count) in sorted_items
    ]

if __name__ == "__main__":
    DATA_PATH = "data/dataset.json"
    OUTPUT_PATH = "output/interest_predictions.json"
    train_loader, _ = load_dialogue_data(DATA_PATH, test_ratio=0, batch_size=8, shuffle=False, transform=None, collate_fn=collate_text, seed=42)
    test_predictions = identify_interest_loader(train_loader)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(test_predictions, f, ensure_ascii=False, indent=4)
