import json, time
from tqdm.asyncio import tqdm  
from openai import OpenAI
from dotenv import load_dotenv
from datasets import load_dataset
import os
from typing import Optional

def get_deepseek_key():
    """Load and return the PRIVATE_KEY from environment."""
    load_dotenv()                  # finds and reads .env automatically
    return os.getenv("DEEPSEEK_KEY")

def call_api(client: OpenAI, system_prompt: str, user_input: str, *, model_name: str = "deepseek-chat", 
             temperature: float = 1.3, max_retries: int = 3,retry_delay: float = 1.0,) -> str:
    """
    Send a single USER message to the chat model and return the response
    text.  Retries up to `max_retries` times on exceptions.

    Returns either the translated text (str) or an error string.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_input},
                ],
                temperature=temperature,
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries:
                return f"Error after {max_retries} attempts: {e}"
            time.sleep(retry_delay)


def call_api_from_dataset(dataset, output_file_name: Optional[str], system_prompt: str,*, 
                          existing_json_path: Optional[str] = None, max_retries: int = 3, retry_delay: float = 1.0):
    """
    Translate the 'text' column of a Hugging Face `datasets.Dataset`
    into Simplified Chinese ('text_zh'), skipping rows whose `id`
    already appear in `existing_json_path` (if provided).

    Returns
    -------
    list[dict]  -- the freshly generated result rows
    """
    # ------------------ duplicate‑filtering ------------------------
    already_done = set()
    if existing_json_path and os.path.exists(existing_json_path):
        with open(existing_json_path, "r", encoding="utf-8") as f:
            already_done = {rec["id"] for rec in json.load(f)}
        print(f"🔍  Loaded {len(already_done):,} IDs already processed.")

    client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")

    new_results, skipped = [], 0

    for row in tqdm(dataset, desc="Processing dataset"):
        record_id = row["id"]
        if record_id in already_done:
            skipped += 1
            continue

        text_en = row["text"]
        labels  = row["labels"]

        if not text_en:
            translation = "No input provided"
        else:
            translation = call_api(
                client,
                system_prompt,
                text_en,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )

        new_results.append({
            "id":      record_id,
            "labels":  labels,
            "text":    text_en,
            "text_zh": translation
        })

    # ------------------ optionally write to disk -------------------
    if output_file_name:
        with open(output_file_name, "w", encoding="utf-8") as f:
            json.dump(new_results, f, indent=4, ensure_ascii=False)
        print(f"✅ Saved {len(new_results):,} new rows to {output_file_name}")

    print(f"   ↪︎ Skipped {skipped:,} duplicate rows")
    return new_results


if __name__ == "__main__":
    DEEPSEEK_KEY = get_deepseek_key()
    
    SYSTEM_PROMPT = (
        "You are a professional translator. "
        "Translate the USER message from English to Simplified Chinese. "
        "Return ONLY the translated Chinese sentence—no explanations, no extra text."
    )

    EXISTING_FILE = "data/goemotions_train_1k_zh.json"

    goemotions = load_dataset("google-research-datasets/go_emotions", "simplified")
    next7k = goemotions["train"].select(range(1000, 8000))

    fresh_rows = call_api_from_dataset(
        next7k,
        output_file_name=None,          # we'll handle merging manually
        system_prompt=SYSTEM_PROMPT,
        existing_json_path=EXISTING_FILE,
    )

    with open(EXISTING_FILE, "r", encoding="utf-8") as f:
        combined = json.load(f)
    combined.extend(fresh_rows)

    with open("data/goemotions_train_8k_zh.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=4, ensure_ascii=False)

    print("📦 File now contains", len(combined), "rows (should be 8 000).")