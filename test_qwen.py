import os
import base64
from openai import OpenAI
from dotenv import load_dotenv
from pdf2image import convert_from_path

load_dotenv(os.path.expanduser("~/.env"))

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

print("Testing Qwen2-VL")

#convert first page to pdf to image
pdf_path = "data/samples/bank_statement_clean_word.pdf"
images = convert_from_path(
    pdf_path, 
    dpi=200,
    poppler_path = "/home/jovyan/.conda/pkgs/poppler-26.05.0-hfdef1ce_3/bin"
)
page_image = images[0]

#save temporily
page_image.save("/tmp/test_page.png", "PNG")

#convert to base64
with open("/tmp/test_page.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

print("Document converted to img, now sending to qwen")

# Simple text test first
response = client.chat.completions.create(
    model="qwen-vl-max",
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
"document_type": "bank statement / invoice / other",
"vendor_or_bank": "name of the institution",
"account_holder": "name of the person",
"total_amount_due": "the total amount as a number",
"currency": "currency symbol or code",
"statement_date": "date if present",
"line_items": [{"description": "...", "amount": ...}]
}
Return only valid JSON, nothing else."""                
                }
            ]
        }
    ]
)

result = response.choices[0].message.content
print(f"\nQwen2-VL extracted:\n{result}")
print("\nDocument test complete")