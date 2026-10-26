from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch
from dataset import load_dialogue_data
import json

# 添加設備設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_inference(loader):
    # label_mapping = {
    #     "LABEL_0": "平淡語氣",
    #     "LABEL_1": "關切語調",
    #     "LABEL_2": "開心語調",
    #     "LABEL_3": "憤怒語調",
    #     "LABEL_4": "悲傷語調",
    #     "LABEL_5": "疑問語調",
    #     "LABEL_6": "驚奇語調",
    #     "LABEL_7": "厭惡語調"
    # }
    label_mapping = {
    "LABEL_0": "Neutral",
    "LABEL_1": "Concerned",
    "LABEL_2": "Happy",
    "LABEL_3": "Angry",
    "LABEL_4": "Sad",
    "LABEL_5": "Questioning",
    "LABEL_6": "Surprised",
    "LABEL_7": "Disgusted"
    }
    
    model_path="Johnson8187/Chinese-Emotion"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)  # 移動模型到設備
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

    results = []
    for texts in loader:                              # texts == List[str] (len ≤ batch_size)
        batch_out = classifier(texts, truncation=True, padding=True)
        for text, out in zip(texts, batch_out):
            results.append(
                dict(
                    text=text,
                    emotion=label_mapping[out["label"]],
                    score=out["score"],
                    raw_label=out["label"],
                )
            )

    return results

def collate_text(batch):
    return [item["dialog_item"]["text"] for item in batch if item["dialog_item"]["role"] in "child"]

if __name__ == "__main__":
    DATA_PATH = "data/dataset.json"
    OUTPUT_PATH = "output/predictions.json"
    train_loader, _ = load_dialogue_data(DATA_PATH, test_ratio=0, batch_size=8, shuffle=False, transform=None, collate_fn=collate_text, seed=42)
    test_predictions = run_inference(train_loader)

    for p in test_predictions:
        print(f"{p['emotion']} ({p['score']:.3f})  —  {p['text']}")

    # Save to JSON
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(test_predictions, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(test_predictions)} predictions to predictions.json")