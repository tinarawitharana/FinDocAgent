import os
import sys
import json
import time
import base64
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from dotenv import load_dotenv
from pdf2image import convert_from_path
from evaluation.metrics import anls_score

load_dotenv(os.path.expanduser("~/.env"))

POPPLER_PATH = "/home/jovyan/.conda/pkgs/poppler-26.05.0-hfdef1ce_3/bin"

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

def pdf_page_to_base64(pdf_path, page_num=0):
    images = convert_from_path(pdf_path, dpi=200, poppler_path=POPPLER_PATH)
    image = images[page_num]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name, "PNG")
        tmp_path = tmp.name
    with open(tmp_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    os.unlink(tmp_path)
    return b64

def run_sec10k_evaluation():
    print("="*60)
    print("SEC 10-K EVALUATION - FinDocAgent")
    print("="*60)

    with open("data/sec10k_qa.json") as f:
        qa_data = json.load(f)

    all_scores = []
    results = []

    for i, item in enumerate(qa_data):
        pdf_path = item["pdf"]
        page_num = item.get("page", 0)
        company = item["company"]
        question = item["question"]
        answers = item["answers"]

        print(f"\n[{i+1}/{len(qa_data)}] {company}")
        print(f"  Q: {question}")
        print(f"  GT: {answers[0]}")

        try:
            image_b64 = pdf_page_to_base64(pdf_path, page_num)

            response = client.chat.completions.create(
                model="qwen-vl-max",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                            },
                            {
                                "type": "text",
                                "text": f"Look at this financial document carefully. {question}\nAnswer with the exact value from the document. Be concise."
                            }
                        ]
                    }
                ],
                max_tokens=50,
                temperature=0
            )

            prediction = response.choices[0].message.content.strip()
            score = max(anls_score(prediction, gt) for gt in answers)
            all_scores.append(score)

            print(f"  Pred: {prediction}")
            print(f"  ANLS: {score:.3f}")

            results.append({
                "company": company,
                "question": question,
                "ground_truths": answers,
                "prediction": prediction,
                "anls": round(score, 3)
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            all_scores.append(0.0)

        time.sleep(0.5)

    avg_anls = sum(all_scores) / len(all_scores) if all_scores else 0

    print(f"\n{'='*60}")
    print(f"SEC 10-K ANLS:    {avg_anls:.3f}")
    print(f"Questions:        {len(all_scores)}")
    print(f"{'='*60}")

    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/sec10k_results.json", "w") as f:
        json.dump({
            "dataset": "SEC 10-K (manual QA)",
            "num_questions": len(all_scores),
            "anls": round(avg_anls, 3),
            "per_question": results
        }, f, indent=2)

    print("Saved to evaluation/results/sec10k_results.json")
    return avg_anls

if __name__ == "__main__":
    run_sec10k_evaluation()
