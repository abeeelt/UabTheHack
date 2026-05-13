#Grupo formado por: Abel, Guilherme, Santi y Uven


#··························
#ESTRUCTURAS DE DATOS
#··························

from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from typing_extensions import TypedDict

class CriterionEvaluation(BaseModel): #Para tomar decisiones. Es lo que sale en el json
    #Primero guardamos el criterio exacto a evaluar sin traducir para que no se líe en la traducción (ya ha pasado)
    criterion_text: str = Field(description="Copia EXACTAMENTE el texto original del criterio en inglés. PROHIBIDO TRADUCIRLO.")

    #Añadimos un paso de pensamiento y verificación --> esto reduce las alucinaciones y mejora el veredicto
    verification_step: str = Field(description="Paso de verificación: Piensa paso a paso, razonamiento estricto. Compara explícitamente los datos (sí/no/valores) del paciente con el criterio.¿Aparece la condición exacta en el texto del paciente? No asumas correlaciones ni pronósticos futuros.")
    
    #Con Literal le decimos al LLM que solo puede decir una de estas tres opciones, nada más
    status: Literal["met", "not_met", "not_enough_info"] = Field(description="Si falta un valor exacto (ej. pide -3dB y solo dice 'moderado'), debe ser 'not_enough_info'.")
    reasoning: str = Field(description="Justificación y evaluación clínica concisa en español.")
    

    #Hacemos que siempre salga la pregunta, cuando poniamos este campo de forma opcional muchas veces se olvidaba y no la formulaba
    missing_clinical_question: str = Field(description="Si el estado es 'not_enough_info', escribe la pregunta exacta al paciente. Si es 'met' o 'not_met', escribe estrictamente 'N/A'.")


class TrialEvaluation(BaseModel): #Agrupamos toda la evaluacion en un ensayo bajo un mismo identificador nct_id
    #Anteriormente el LLM calculaba el score pero lo dejamos en calculos con Python para no saturar al LLm y seguir la misma logica siempre
    nct_id: str
    criteria: List[CriterionEvaluation]

class PatientExtraction(BaseModel): #extrae los conceptos clave del texto
    #Cogemos el término clave de ClinicalTrials
    primary_condition: str = Field(description="Enfermedad principal para buscar en ClinicalTrials.gov (en inglés)")
    mesh_terms: List[str] = Field(description="Lista de términos MeSH") #Para usar los términos médicos y no usar por ejemplo sinónimos

class StructuredPatient(BaseModel): #Esto lo hacemos para que no entre la informacion del paciente como un texto grande sin estructurar, sino que creamos un perfil basico y lo simplificado con estos campos
    age: Optional[int] = Field(None, description="Edad del paciente en años.")
    gender: Optional[str] = Field(None, description="Género del paciente.")
    medical_conditions: List[str] = Field(description="Enfermedades actuales o pasadas en español.")
    clinical_measurements: List[str] = Field(description="Valores médicos en español (ej. visión 0.3, presión, etc).")
    other_info: List[str] = Field(description="Historial quirúrgico (sí/no), alergias u otros datos en español.")

class AgentState(TypedDict): #Por ultimo para los estados del grafo con LangGraph definimos su memoria y su flujo
    patient_profile: str  #Texto grande sin estructurar
    structured_patient: str #Perfil creado con StructuredPatient
    mesh_terms: List[str] #Para los terminos MeSH y mejorar la busqueda
    retrieved_trials: List[dict] #Lista de ensayos de la api de Clinical
    evaluated_trials: List[dict] # lista de diccionarios preparados para el ranking y el JSON
