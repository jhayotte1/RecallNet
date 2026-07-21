from pydantic import BaseModel, Field
from typing import Literal


class TripleEvaluation(BaseModel):
    reasoning: str = Field(description="Only one sentence max")
    meaningfulness: int = Field(ge=0, le=5)
    typicality : int = Field(ge=0, le=5)
    saliency: int = Field(ge=0, le=5)

class BatchEvaluation(BaseModel):
    evaluations: dict[str, TripleEvaluation]


#For rule-mining
class TripleReview(BaseModel):
    reasoning: str = Field(description= "One sentence max")
    verdict: str = Field(description= "KEEP, REJECT, or UNCERTAIN")

class BatchReview(BaseModel):
    reviews: dict[str, TripleReview]


#Second filtering
class TripleFilterDecision(BaseModel):
    reasoning: str = Field(description="One sentence max")
    decision: Literal["KEEP", "MODIFY", "REJECT"]

class BatchFilterDecision(BaseModel):
    filterdecision: dict[str, TripleFilterDecision]
