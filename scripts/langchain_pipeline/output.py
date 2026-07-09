from pydantic import BaseModel, Field
from typing import Literal


class TripleEvaluation(BaseModel):
    reasoning: str = Field(description="Only one sentence max")
    meaningfulness: int = Field(ge=0, le=5)
    typicality : int = Field(ge=0, le=5)
    saliency: int = Field(ge=0, le=5)

class BatchEvaluation(BaseModel):
    evaluations: dict[str, TripleEvaluation]