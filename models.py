from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class CriterionEvaluation(BaseModel):
    criterion_text: str = Field(description="Texto exacto del criterio evaluado")
    status: Literal["met", "not met", "not enough info"] = Field(description="Evaluación estricta del criterio")
    reasoning: str = Field(description="Justificación clínica breve de la decisión")
    missing_clinical_question: Optional[str] = Field(None, description="Pregunta exacta al paciente si falta información")

class TrialEvaluation(BaseModel):
    nct_id: str
    criteria: List[CriterionEvaluation]
    is_eligible: bool = Field(description="True SOLO si NO hay ningún criterio 'not met'")
    score: float = Field(description="Score para ranking NDCG@10")

# LangGraph prefiere TypedDict o Pydantic para el estado. Usaremos TypedDict para el StateGraph.
from typing_extensions import TypedDict

class AgentState(TypedDict):
    patient_profile: str
    mesh_terms: List[str]
    retrieved_trials: List[dict]
    evaluated_trials: List[TrialEvaluation]
