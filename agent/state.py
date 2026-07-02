from typing import TypedDict, List, Optional

class AgentState(TypedDict):

    #document analyzing
    document_path: str

    #quesion to answer
    question: str

    #retriver finds
    retrieved_chunks: List[str]

    #extractor pulls out
    extracted_fields: dict

    #agents answer
    answer: str

    #anamoly checker flags
    anomalies: List[str]

    #final output
    risk_report: str

    #control
    iteration_count: int
    task_complete: bool