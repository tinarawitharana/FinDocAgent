"""Shared state passed between all LangGraph nodes in the FinDocAgent pipeline."""

from typing import TypedDict, List, Optional

class AgentState(TypedDict):

    # path to the input document (PDF or image)
    document_path: str

    # question to answer, if in document-QA/RAG mode; unset in bank-statement mode
    question: str

    # retriever output: either rendered page-image paths (RAG mode) or the raw
    # document_path passed through unchanged (bank-statement / image mode)
    retrieved_chunks: List[str]

    # extractor output in bank-statement mode: structured fields pulled from the document
    extracted_fields: dict

    # extractor output in document-QA/RAG mode: the model's answer to `question`
    answer: str

    # anomaly_checker output: list of anomaly dicts (math discrepancies, date ordering, ...)
    anomalies: List[str]

    # explainer output: the final human-readable risk report
    risk_report: str

    # loop control
    iteration_count: int
    task_complete: bool