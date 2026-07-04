from agent.state import AgentState
from models.qwen import extract_fields_from_document
from openai import OpenAI
from dotenv import load_dotenv
import base64
import os

load_dotenv(os.path.expanduser("~/.env"))

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

)

def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extractor_node(state: AgentState) -> AgentState:
    print(f"[EXTRACTOR] Extracting fields using Qwen2-VL...")

    #docvqa - answer specific question
    if state.get("question"):
        image_path = state["retrieved_chunks"][0]
        print(f"[EXTRACTOR] Question mode: {state['question']}")

        image_b64 = image_to_base64(image_path)

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
                            "text": f"Look at the document carefully. Find the section relevant to this question: {state['question']}\nExtract the exact answer as it appears in the document. Be concise, one word or short phrase only."



                        }
                    ]
                }
            ],
            max_tokens=50,
            temperature =0
           
        )

        state["answer"] = response.choices[0].message.content.strip()
        state["task_complete"] = True
        print(f" [EXTRACTOR] Answer: {state['answer']}")

        return state

    #pdf mode
    document_path = state["document_path"]

    #use qwen to extract strcutures fields
    extracted = extract_fields_from_document(document_path)

    #map to standard format
    extracted_fields = {
        "vendor": extracted.get("vendor_or_bank", "Unknown"),
        "account_holder": extracted.get("account_holder", "Unknown"),
        "document_type": extracted.get("document_type", "Unknown"),
        "stated_total": float(extracted.get("stated_total", 0)),
        "currency": extracted.get("currency", "Unknown"),
        "line_items": extracted.get("line_items", []),
        "statement_date": extracted.get("statement_date", None)

    }

    state["extracted_fields"] = extracted_fields 

    print(f"[EXTRACTOR] Vendor: {extracted_fields['vendor']}")
    print(f"[EXTRACTOR] Stated total: {extracted_fields['currency']}{extracted_fields['stated_total']}")
    print(f"[EXTRACTOR] Line items found: {len(extracted_fields['line_items'])}")
    
    return state

