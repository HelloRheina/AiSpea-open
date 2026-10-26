import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

model_name = "Qwen/Qwen1.5-1.8B-Chat"

# Load Qwen model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True).eval()

# 中文分类 Prompt
SYSTEM_PROMPT = """
你是一位擅长儿童教育分析的助手。

请根据小朋友的中文对话，判断出他的兴趣类型。核心兴趣领域包括：
1. STEM（科学、技术、工程、数学）
2. 语言与读写
3. 艺术
4. 社会研究

然后再进一步给出一个子兴趣，如：生物、天文、机器人、绘画、写作等。

最终输出格式如下，请严格使用 JSON，不要多说一句话：

{
  "core_interest": "STEM",
  "sub_interest": "生物"
}
"""

def classify_interest_qwen_chinese(dialog: str) -> dict:
    # Format prompt as Chat-style
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": dialog}
    ]
    
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, max_new_tokens=256, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract JSON block from output
    try:
        json_start = response.index('{')
        json_end = response.rindex('}') + 1
        parsed = json.loads(response[json_start:json_end])
        return parsed
    except Exception as e:
        print("解析失败:", e)
        print("原始回复:", response)
        return {"core_interest": "未知", "sub_interest": "未识别"}

# 🔍 Test Dialogs
dialogs = [
    "我喜欢看蚂蚁搬东西，它们好聪明！",
    "我画了一只独角兽，还编了一个故事。",
    "我想造一个桥，让玩具车过河。",
    "我喜欢看书，也喜欢讲故事给朋友听。",
    "我想做一个机器人帮我扫地。"
]

for i, dialog in enumerate(dialogs):
    result = classify_interest_qwen_chinese(dialog)
    print(f"对话 {i+1}: {result}")
