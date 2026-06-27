from agent.state import AgentState
from models.qwen import extract_fields_from_document



def extractor_node(state: AgentState) -> AgentState:
    print(f"[EXTRACTOR] Extracting fields using Qwen2-VL...")

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

