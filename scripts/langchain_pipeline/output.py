from pydantic import BaseModel, Field
from typing import Literal


class TripleClassificaiton(BaseModel):
    """A triple, its label and reasoning for this label"""
    reasoning: str = Field(description="Only one sentence max")
    label: Literal["VALID", "NOISY"]

class BatchClassification(BaseModel):
    classifications: dict[str, TripleClassificaiton]