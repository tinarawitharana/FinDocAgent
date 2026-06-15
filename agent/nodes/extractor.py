from agent.state import AgentState
import re

def extract_amounts(text):

    #finds all money like numbers in text

    #pattern 
    pattern = r'\d{1,3}(?:,\d{3})*\.\d{2}'
    matches = re.findall(pattern, text)

    #convert to float
    amounts = [float(match.replace(',', '')) for match in matches]

    return amounts

def extract_dates(text):

    #finds date like patterns in text

    pattern = r'\d{1,2}\s[A-Za-z]{3,9}\s+\d{2,4}'
    matches = re.findall(pattern, text)

    return matches

def extract_stated_total(text):

    #looks for a number that appears near the word 'Total'
    pattern = r'Total[^£\d]*(\d{1,3}(?:,\d{3})*\.\d{2})'
    matches = re.findall(pattern, text, re.IGNORECASE)

    if matches:
        return float(matches[0].replace(",", ""))
    return None

def extractor_node(state: AgentState) -> AgentState:
    print(f"[EXTRACTOR] Extracting fields from retrieved chunks...")

    #combiene all retrieved chunks into a single text
    full_text = " ".join(state["retrieved_chunks"])


    #find all amounts and dates in the text
    all_amounts = extract_amounts(full_text)
    all_dates = extract_dates(full_text)

    #better approach
    stated_total = extract_stated_total(full_text)

    if stated_total is None:
        stated_total = max(all_amounts) if all_amounts else 0
        print("[EXTRACTOR] No 'Total' label found - move back to largest amount")


    #the rest of the amounts are line items
    other_amounts = [a for a in all_amounts if a != stated_total]
    line_items = [{"description": "extracted_item", "amount": a} for a in other_amounts]

    extracted_fields = {
        "vendor": "Unknown (doesnt identify names)",   #dummy
        "stated_total": stated_total,
        "line_items": line_items,
        "dates_found": all_dates,
        "all_amounts_found": all_amounts
    }

    state['extracted_fields'] = extracted_fields

    print(f"[EXTRACTOR] FOUND {len(all_amounts)} amounts, {len(all_dates)} dates")
    print(f"[EXTRACTOR] Stated total: £{stated_total}, Line items: {len(line_items)}")

    return state

