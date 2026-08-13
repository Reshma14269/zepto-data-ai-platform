"""
Pydantic models for the /ask endpoint.

AskRequest  -> incoming request body: {"query": "..."}
AskResponse -> the structured-output guarantee described in the module spec:
    answer     (str)   the generated answer text
    sources    (list)  chunk/document ids used to ground the answer.
                        Empty for general_question (no retrieval happened).
    confidence (float) 0-1 confidence score.
"""

from typing import List
from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question.")


class AskResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("answer")
    @classmethod
    def answer_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("answer must not be empty")
        return v
