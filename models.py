from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class CriterionEvaluation(BaseModel):
    criterion_text: str
    status: Literal["met", "not met", "not enough info"] = Field(description="Evaluación estricta")
    reasoning: str
    missing_clinical_question: Optional[str] = Field(None, description="Pregunta si falta info")

class TrialEvaluation(BaseModel):
    nct_id: str
    criteria: List[CriterionEvaluation]
    is_eligible: bool
    score: float = Field(description="Score para NDCG@10")

class AgentState(BaseModel):
    patient_profile: str
    mesh_terms: List[str] = []
    retrieved_trials: List[dict] = []
    evaluated_trials: List[TrialEvaluation] = []
