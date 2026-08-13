"""RQ3 explainability: generates paired attention-map visualizations for SEC 10-K
questions at both the RAG-retrieved page and the known-correct ("oracle") page, so a
human rater can compare where the model actually looked in each case
(see evaluation/human_eval_rater_instructions.md and human_eval_scoring_template.csv)."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
from models.attention_map import generate_attention_map

# Override via the POPPLER_PATH env var if poppler isn't on your PATH.
POPPLER_PATH = os.getenv("POPPLER_PATH")

SEC10K_QUESTION_NUMBERS = [1, 2, 4, 5, 8, 14, 15, 17, 19, 22, 26, 28, 30, 36, 37]

# question_number -> top-ranked page RAG actually retrieved (from the last full eval run)
RAG_RETRIEVED_PAGE = {
    2: 116,
    14: 86,
    15: 266,
    17: 266,
    22: 23,
    28: 265,
    36: 92,
}

BANK_STATEMENTS = [
    "data/samples/synthetic/statement_01_clean.pdf",
    "data/samples/synthetic/statement_02_math_error.pdf",
    "data/samples/synthetic/statement_03_date_error.pdf",
    "data/samples/synthetic/statement_04_both_errors.pdf",
    "data/samples/bs1.pdf",
    "data/samples/bs2.pdf",
    "data/samples/bs3.pdf",
    "data/samples/bs4.pdf",
    "data/samples/bs5.pdf",
    "data/samples/bank_statement_anomaly_word.pdf",
]

BANK_STATEMENT_QUESTION = "What is the total amount stated on this bank statement, and are the transaction dates in chronological order?"


def run_sec10k_attention():
    with open("data/sec10k_qa.json") as f:
        qa_data = json.load(f)

    os.makedirs("evaluation/results/attention_maps/sec10k", exist_ok=True)

    for q_num in SEC10K_QUESTION_NUMBERS:
        item = qa_data[q_num - 1]
        pdf_path = item["pdf"]
        question = item["question"]

        # oracle page (always run)
        oracle_page = item["page"]
        print(f"\n[SEC10K Q{q_num}] oracle page {oracle_page}: {question}")
        imgs = convert_from_path(pdf_path, first_page=oracle_page, last_page=oracle_page, dpi=150, poppler_path=POPPLER_PATH)
        save_path = f"evaluation/results/attention_maps/sec10k/q{q_num}_oracle.png"
        try:
            answer, fig = generate_attention_map(imgs[0], question, save_path=save_path)
            print(f"  Oracle answer: {answer}")
            if fig:
                plt.close(fig)
        except Exception as e:
            print(f"  ERROR (oracle): {e}")

        # RAG-retrieved page (only for the failure cases)
        if q_num in RAG_RETRIEVED_PAGE:
            rag_page = RAG_RETRIEVED_PAGE[q_num]
            print(f"[SEC10K Q{q_num}] RAG-retrieved page {rag_page}")
            imgs = convert_from_path(pdf_path, first_page=rag_page, last_page=rag_page, dpi=150, poppler_path=POPPLER_PATH)
            save_path = f"evaluation/results/attention_maps/sec10k/q{q_num}_rag.png"
            try:
                answer, fig = generate_attention_map(imgs[0], question, save_path=save_path)
                print(f"  RAG answer: {answer}")
                if fig:
                    plt.close(fig)
            except Exception as e:
                print(f"  ERROR (rag): {e}")


def run_bank_statement_attention():
    os.makedirs("evaluation/results/attention_maps/bank_statements", exist_ok=True)

    for pdf_path in BANK_STATEMENTS:
        doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"\n[Bank Statement] {doc_name}")

        imgs = convert_from_path(pdf_path, dpi=150, poppler_path=POPPLER_PATH)
        save_path = f"evaluation/results/attention_maps/bank_statements/{doc_name}.png"
        try:
            answer, fig = generate_attention_map(imgs[0], BANK_STATEMENT_QUESTION, save_path=save_path)
            print(f"  Answer: {answer}")
            if fig:
                plt.close(fig)
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    run_sec10k_attention()
    run_bank_statement_attention()
