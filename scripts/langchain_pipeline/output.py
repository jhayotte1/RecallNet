from pydantic import BaseModel, Field
from typing import Litteral


class TripleClassificaiton(BaseModel):
    """A triple, its label and reasoning for this label"""
    reasoning: str = Field(description="Only one sentence max")
    label: str = Litteral["VALID", "NOISY"]

class BatchClassification(BaseModel):
    classifications: dict[str, TripleClassificaiton]