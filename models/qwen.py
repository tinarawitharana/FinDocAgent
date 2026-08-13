"""Qwen2-VL-max field extraction for financial documents (bank statements, invoices, KYC docs).

Calls the DashScope-hosted Qwen2-VL-max model with a rendered page image and a fixed
extraction prompt, returning structured JSON (totals, line items, balances).
"""

import os
import base64
import json
from openai import OpenAI
from dotenv import load_dotenv
from pdf2image import convert_from_path

load_dotenv(os.path.expanduser("~/.env"))

# poppler is a system dependency for pdf2image. Override via the POPPLER_PATH env var
# if it's not on your PATH (e.g. a conda-installed poppler); leave unset to use PATH.
POPPLER_PATH = os.getenv("POPPLER_PATH")

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

)

def pdf_page_to_base64(pdf_path, page_num=0, dpi=200):
    """Render one page of a PDF to a base64-encoded PNG for the vision API."""
    images = convert_from_path(
        pdf_path,
        dpi=dpi,
        poppler_path = POPPLER_PATH
    )

    image = images[page_num]
    image.save("/tmp/qwen_page.png", "PNG")
    with open("/tmp/qwen_page.png", "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def extract_fields_from_document(pdf_path):
    """Send the document's first page to Qwen2-VL-max and parse its JSON field extraction.

    Returns a dict with the extracted fields, or a fallback dict with the raw response
    under "raw_response" if the model's output isn't valid JSON.
    """
    print(f"[QWEN] Converting document to image..")
    image_b64 = pdf_page_to_base64(pdf_path, page_num=0)

    print(f"[QWEN] Sending to Qwen2-vl for extraction...")
    response = client.chat.completions.create(
        model = "qwen-vl-max",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                            }
                    },
                    {
                        "type": "text",
                        "text": """You are a financial document analyst. Extract the following fields from this document and return them as JSON:
{
"document_type": "bank statement / invoice / kyc / other",
"vendor_or_bank": "name of the institution or vendor",
"account_holder": "name of the person or company",
"stated_total": <total amount as a number only, no currency symbol>,
"currency": "currency symbo, or code",
"statement_date": "date if present, else null",
"opening_balance": <the starting/opening balance shown in the document's header or summary, as a positive number. Use 0 if this document has no running balance (e.g. a credit card "total amount due" style statement with no opening balance concept)>,
"line_items": [{"description": "...", "amount": <the number as printed, always positive>, "date": "YYYY-MM-DD if present, else null", "type": "credit if this is money in / paid in / a deposit, debit if this is money out / paid out / a withdrawal. If the document has separate Credit and Debit (or Paid In and Paid Out) columns, use whichever column this row's amount appears in. Do NOT include an opening balance or balance-brought-forward row here - put that value in the separate opening_balance field instead."}]
}

Return only valid JSON, nothing else."""


                    }
                ]
            }
        ]
    )

    result_text = response.choices[0].message.content

    #clean up makdown code 
    result_text = result_text.strip()
    if result_text.startswith("'''"):
        result_text = result_text.split("'''")[1]
        if result_text.startswith("json"):
            result_text = result_text[4:]

    try:
        extracted = json.loads(result_text)
        print(f"[QWEN] Extraction successful")
        return extracted
    except json.JSONDecodeError:
        print(f"[QWEN] JSON parse error - returning raw text")
        return {"raw_response": result_text, "stated_total":0, "line_items": []}
