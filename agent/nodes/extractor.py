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

    #question answering mode (image or PDF via RAG)
    if state.get("question"):
        chunk = state["retrieved_chunks"][0]
        print(f"[EXTRACTOR] Question mode: {state['question']}")

        if os.path.isfile(chunk):
            content = []
            for image_path in state["retrieved_chunks"]:
                image_b64 = image_to_base64(image_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                })

            content.append({
                "type": "text",
                "text": (
                    f"Look at these document pages carefully. Find the section relevant to this question: {state['question']}\n"
                    "These pages may contain tables with multiple business segments and multiple fiscal years side by side. "
                    "Before answering, briefly identify: (1) the exact row/line-item label that matches the question, "
                    "(2) which segment/column it belongs to if the table has more than one segment, and "
                    "(3) which fiscal year column it belongs to (use fiscal year 2023 unless the question asks otherwise).\n"
                    "If you cannot clearly find this exact row, segment, and year in the visible pages, do not guess — "
                    "state that identification step as 'not confident' and give your final answer as 'Not found'.\n"
                    "State that identification in one short line, then on a new line write your final answer prefixed exactly with 'ANSWER: '. "
                    "The final answer must be the exact number or short phrase as it appears in the document, with no added currency symbols, "
                    "units, or words that are not already printed next to it — do not round or convert units."
                )
            })


            response = client.chat.completions.create(
                model="qwen-vl-max",
                messages=[{"role": "user", "content": content}],
                max_tokens=200,
                temperature=0
            )


            raw_output = response.choices[0].message.content.strip()
            if "ANSWER:" in raw_output:
                state["answer"] = raw_output.split("ANSWER:")[-1].strip()
            else:
                state["answer"] = raw_output

        
        print(f"[EXTRACTOR] Answer: {state['answer']}")

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

