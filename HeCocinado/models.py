from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from typing_extensions import TypedDict

class CriterionEvaluation(BaseModel):
    criterion_text: str = Field(description="Copia EXACTAMENTE el texto original del criterio en inglés. PROHIBIDO TRADUCIRLO.")
    verification_step: str = Field(description="Paso de verificación: Piensa paso a paso, razonamiento estricto. Compara explícitamente los datos (sí/no/valores) del paciente con el criterio.¿Aparece la condición exacta en el texto del paciente? No asumas correlaciones ni pronósticos futuros.")
    status: Literal["met", "not_met", "not_enough_info"] = Field(description="Si falta un valor exacto (ej. pide -3dB y solo dice 'moderado'), debe ser 'not_enough_info'.")
    reasoning: str = Field(description="Justificación y evaluación clínica concisa en español.")
    missing_clinical_question: str = Field(description="Si el estado es 'not_enough_info', escribe la pregunta exacta al paciente. Si es 'met' o 'not_met', escribe estrictamente 'N/A'.")


class TrialEvaluation(BaseModel):
    # ¡HEMOS QUITADO EL SCORE Y EL IS_ELIGIBLE DE AQUÍ! El LLM solo evalúa criterios.
    nct_id: str
    criteria: List[CriterionEvaluation]

class PatientExtraction(BaseModel):
    primary_condition: str = Field(description="Enfermedad principal para buscar en ClinicalTrials.gov (en inglés)")
    mesh_terms: List[str] = Field(description="Lista de términos MeSH")

class StructuredPatient(BaseModel):
    age: Optional[int] = Field(None, description="Edad del paciente en años.")
    gender: Optional[str] = Field(None, description="Género del paciente.")
    medical_conditions: List[str] = Field(description="Enfermedades actuales o pasadas en español.")
    clinical_measurements: List[str] = Field(description="Valores médicos en español (ej. visión 0.3, presión, etc).")
    other_info: List[str] = Field(description="Historial quirúrgico (sí/no), alergias u otros datos en español.")

class AgentState(TypedDict):
    patient_profile: str
    structured_patient: str
    mesh_terms: List[str]
    retrieved_trials: List[dict]
    evaluated_trials: List[dict] # AHORA ES UNA LISTA DE DICCIONARIOS