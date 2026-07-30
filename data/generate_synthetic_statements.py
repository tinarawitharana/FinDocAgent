import os
import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch


def generate_bank_statement(output_path, bank_name, account_holder, transactions,
                              inject_math_error=0.0, swap_dates=None):
    """
    transactions: list of dicts [{"date": "2024-01-05", "description": "...", "amount": 120.50}, ...]
    inject_math_error: amount added to the stated total so it no longer matches the sum of transactions
    swap_dates: tuple of two indices (i, j) whose dates get swapped, creating an out-of-order anomaly
    """
    txns = [dict(t) for t in transactions]  # copy so we don't mutate the caller's list

    if swap_dates:
        i, j = swap_dates
        txns[i]["date"], txns[j]["date"] = txns[j]["date"], txns[i]["date"]

    calculated_total = sum(t["amount"] for t in txns)
    stated_total = round(calculated_total + inject_math_error, 2)

    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(f"<b>{bank_name}</b>", styles["Title"]))
    elements.append(Paragraph(f"Account Holder: {account_holder}", styles["Normal"]))
    elements.append(Paragraph(f"Statement Total: {stated_total:.2f}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    table_data = [["Date", "Description", "Amount"]]
    for t in txns:
        table_data.append([t["date"], t["description"], f"{t['amount']:.2f}"])

    table = Table(table_data, colWidths=[1.3 * inch, 3.5 * inch, 1.3 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    elements.append(table)

    doc.build(elements)

    return {
        "file": output_path,
        "bank_name": bank_name,
        "account_holder": account_holder,
        "stated_total": stated_total,
        "calculated_total": calculated_total,
        "math_discrepancy": round(stated_total - calculated_total, 2),
        "date_swap_applied": swap_dates is not None
    }


def main():
    base_transactions = [
        {"date": "2024-01-01", "description": "Opening Balance", "amount": 500.00},
        {"date": "2024-01-03", "description": "Grocery Store", "amount": -45.20},
        {"date": "2024-01-05", "description": "Salary Payment", "amount": 2000.00},
        {"date": "2024-01-08", "description": "Electricity Bill", "amount": -80.00},
        {"date": "2024-01-10", "description": "Restaurant", "amount": -35.50},
        {"date": "2024-01-12", "description": "Transfer In", "amount": 150.00},
        {"date": "2024-01-15", "description": "Gym Membership", "amount": -30.00},
    ]

    os.makedirs("data/samples/synthetic", exist_ok=True)
    ground_truth = []

    # 1. Fully clean — control case
    ground_truth.append(generate_bank_statement(
        "data/samples/synthetic/statement_01_clean.pdf",
        "Metro Bank", "Alex Carter", base_transactions
    ))

    # 2. Math discrepancy only
    ground_truth.append(generate_bank_statement(
        "data/samples/synthetic/statement_02_math_error.pdf",
        "Metro Bank", "Alex Carter", base_transactions,
        inject_math_error=250.00
    ))

    # 3. Date-ordering anomaly only
    ground_truth.append(generate_bank_statement(
        "data/samples/synthetic/statement_03_date_error.pdf",
        "Metro Bank", "Alex Carter", base_transactions,
        swap_dates=(2, 4)
    ))

    # 4. Both anomalies together
    ground_truth.append(generate_bank_statement(
        "data/samples/synthetic/statement_04_both_errors.pdf",
        "Metro Bank", "Alex Carter", base_transactions,
        inject_math_error=-120.00, swap_dates=(1, 5)
    ))

    with open("data/samples/synthetic/ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Generated {len(ground_truth)} synthetic statements + ground_truth.json")


if __name__ == "__main__":
    main()
