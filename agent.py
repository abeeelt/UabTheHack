from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from models import TrialEvaluation, CriterionEvaluation

# Usamos Llama3 con temperatura 0 para máxima consistencia
llm = ChatOllama(model="llama3", temperature=0, format="json")

def evaluate_eligibility_node(state: AgentState) -> AgentState:
    print("\n[Nodo 3] Verificación de elegibilidad criterio a criterio...")
    patient = state.patient_profile
    trials = state.retrieved_trials
    evaluated_list = []

    parser = JsonOutputParser(pydantic_object=TrialEvaluation)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un evaluador de ensayos clínicos. Tu tarea es comparar el perfil del paciente con los criterios de inclusión/exclusión.
        Para cada criterio, responde estrictamente:
        - "met": El paciente CUMPLE el criterio.
        - "not met": El paciente NO cumple el criterio.
        - "not enough info": Falta información en el perfil.
        
        Si marcas "not enough info", debes redactar la 'missing_clinical_question' que el médico debería hacerle al paciente.
        Responde SIEMPRE en formato JSON siguiendo el esquema proporcionado."""),
        ("user", "PACIENTE: {patient}\n\nENSAYO (Criterios): {criteria_text}")
    ])

    # Por tiempo de hackathon, evaluamos solo los top 5 resultados de la API
    for study in trials[:5]:
        nct_id = study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
        criteria_text = study.get("protocolSection", {}).get("eligibilityModule", {}).get("eligibilityCriteria", "")

        chain = prompt | llm | parser
        try:
            # Invocación al modelo local
            result = chain.invoke({
                "patient": patient,
                "criteria_text": criteria_text
            })
            
            # Aquí calculamos un score simple para el ranking (Tarea 3)
            # Ejemplo: % de criterios 'met' vs total
            total = len(result['criteria'])
            met_count = sum(1 for c in result['criteria'] if c['status'] == 'met')
            result['score'] = met_count / total if total > 0 else 0
            result['nct_id'] = nct_id
            
            evaluated_list.append(result)
            print(f"   - {nct_id} evaluado. Score: {result['score']}")
        except Exception as e:
            print(f"   - Error evaluando {nct_id}: {e}")

    return {"evaluated_trials": evaluated_list}
