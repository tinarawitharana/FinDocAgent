import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.qwen import extract_fields_from_document

BANK_STATEMENT_PDFS = [
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

def extract_all():
    results = {}
    for pdf_path in BANK_STATEMENT_PDFS:
        print(f"\nExtracting {pdf_path}...")
        fields = extract_fields_from_document(pdf_path)
        results[pdf_path] = fields

    with open("data/samples/bank_statements_extracted.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to data/samples/bank_statements_extracted.json")
    print("Review this file and correct any obviously wrong values before running SHAP.")

if __name__ == "__main__":
    extract_all()
