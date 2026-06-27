import sys
import os
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import anls_score, calculate_f1
from models.qwen import extract_fields_from_document

#manually defined cases
#using the 2 sample docs with known ground truth

TEST_CASES = [
    {
        "document": "data/samples/bank_statement_clean_word.pdf",
        "document_name": "SunTrust Bank Statement (Clean)",
        "questions": [
            {
                "question": "What is the total amount due?",
                "ground_truth": "3898.57",
                "field": "stated_total"
            },
            {
                "question": "Who is the account holder?",
                "ground_truth": "john smith",
                "field": "account_holder"
            },
            {
                "question": "What is the bank name?",
                "ground_truth": "suntrust",
                "field": "vendor"
            }
        ],
        "expected_anomalies": 0  # clean document
    },
    {
        "document": "data/samples/bank_statement_anomaly_word.pdf",
        "document_name": "HSBC Bank Statement (Anomaly)",
        "questions": [
            {
                "question": "What is the closing balance?",
                "ground_truth": "2139.27",
                "field": "stated_total"
            },
            {
                "question": "What is the bank name?",
                "ground_truth": "hsbc",
                "field": "vendor"
            }
        ],
        "expected_anomalies": 1  # has discrepancy
    }
]

def run_regex_baseline(document_path):

    #run regex based extraction 

    import re
    import pdfplumber

    with pdfplumber.open(document_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + " "

    #find amounts
    pattern = r'\d{1,3}(?:,\d{3})*\.\d{2}'
    matches = re.findall(pattern, full_text)
    amounts = [float(m.replace(",", "")) for m in matches]

    #find total near "Total" label
    total_pattern = r'Total[^£\d]*(\d{1,3}(?:,\d{3})*\.\d{2})'
    total_matches = re.findall(total_pattern, full_text, re.IGNORECASE)
    stated_total = float(total_matches[0].replace(",", "")) if total_matches else (max(amounts) if amounts else 0)

    return {
        "vendor": "Unknown (regex)",
        "account_holder": "Unknown (regex)",
        "stated_total": stated_total,
        "line_items": [{"description": "item", "amount": a} for a in amounts if a != stated_total]
    }

def run_qwen_baseline(document_path):
    
    #singel pass Qwen2 vl extraction, vlm baseline, no agent loop

    return extract_fields_from_document(document_path)

    
def evaluate_extraction(extracted, test_case):
    
    #evaluate extraction results against ground truth questions and return anls scores

    scores = []
    results = []

    for qa in test_case["questions"]:
        field = qa["field"]
        ground_truth = qa["ground_truth"]

        #get prediction from extracted fields
        prediction = extracted.get(field, "")
        if prediction is None:
            prediction = ""

        score = anls_score(str(prediction), ground_truth)
        scores.append(score)

        results.append({
            "question": qa["question"],
            "ground_truth": ground_truth,
            "prediction": str(prediction),
            "anls": round(score, 3)
            
        })

    return scores, results

    
def run_evaluation():

    print("=" * 60)
    print("FINDOCAGENT EVALUATION")
    print("=" * 60)

    all_regex_scores = []
    all_qwen_scores = []

    for test_case in TEST_CASES:
        doc = test_case["document"]
        print(f"\n{'='*60}")
        print(f"Document: {test_case['document_name']}")
        print(f"{'='*60}")

        # Check document exists
        if not os.path.exists(doc):
            print(f"[SKIP] Document not found: {doc}")
            continue

        # REGEX BASELINE
        print("\n[1] Running Regex Baseline...")
        start = time.time()
        try:
            regex_extracted = run_regex_baseline(doc)
            regex_time = time.time() - start
            regex_scores, regex_results = evaluate_extraction(
                regex_extracted, test_case
            )

            all_regex_scores.extend(regex_scores)
            print(f"    Time: {regex_time:.2f}s")
            print(f"    Extracted total: {regex_extracted.get('stated_total', 'N/A')}")
            for r in regex_results:
                print(f"    Q: {r['question']}")
                print(f"       GT: {r['ground_truth']} | Pred: {r['prediction']} | ANLS: {r['anls']}")
        except Exception as e:
            print(f"    ERROR: {e}")
            regex_scores = [0.0] * len(test_case["questions"])
            all_regex_scores.extend(regex_scores)

            
        #QWEN2-VL SINGLE PASS

        print("\n[2] Running Qwen2-VL Single-Pass Baseline...")
        start = time.time()
        try:
            qwen_extracted = run_qwen_baseline(doc)
            qwen_time = time.time() - start

            #Normalise field names
            if "vendor_or_bank" in qwen_extracted:
                qwen_extracted["vendor"] = qwen_extracted["vendor_or_bank"]

            qwen_scores, qwen_results = evaluate_extraction(
                qwen_extracted, test_case
            )

            all_qwen_scores.extend(qwen_scores)
            print(f"    Time: {qwen_time:.2f}s")
            print(f"    Extracted total: {qwen_extracted.get('stated_total', 'N/A')}")
            for r in qwen_results:
                print(f"    Q: {r['question']}")
                print(f"       GT: {r['ground_truth']} | Pred: {r['prediction']} | ANLS: {r['anls']}")
        except Exception as e:
            print(f"    ERROR: {e}")
            qwen_scores = [0.0] * len(test_case["questions"])
            all_qwen_scores.extend(qwen_scores)

    #print
    print(f"\n{'='*60}")
    print("SUMMARY RESULTS")
    print(f"{'='*60}")

    avg_regex = sum(all_regex_scores) / len(all_regex_scores) if all_regex_scores else 0
    avg_qwen = sum(all_qwen_scores) / len(all_qwen_scores) if all_qwen_scores else 0

    print(f"\n{'Method':<30} {'Avg ANLS':>10}")
    print(f"{'-'*42}")
    print(f"{'Regex Baseline':<30} {avg_regex:>10.3f}")
    print(f"{'Qwen2-VL Single-Pass':<30} {avg_qwen:>10.3f}")
    print(f"\nImprovement: +{(avg_qwen - avg_regex):.3f} ANLS points")
    print(f"{'='*60}")

    # Save results
    results_data = {
        "regex_avg_anls": round(avg_regex, 3),
        "qwen_avg_anls": round(avg_qwen, 3),
        "improvement": round(avg_qwen - avg_regex, 3)
    }

    os.makedirs("evaluation/results", exist_ok = True)
    with open("evaluation/results/baseline_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to evaluation/results/baseline_results.json")


if __name__ == "__main__":
    run_evaluation()




    

    

