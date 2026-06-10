from typing import TypedDict, List, Optional

class AgentState(TypedDict):

    #document analyzing
    document_path: str

    #retriver finds
    retrieved_chunks: List[str]

    #extractor pulls out
    extracted_fields: dict

    #anamoly checker flags
    anamolies: List[str]

    #final output
    risk_report: str

    #control
    iteration_count: int
    task_complete: bool