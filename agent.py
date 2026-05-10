# agent.py
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from models import AgentState, TrialEvaluation, PatientExtraction
from api_tools import fetch_clinical_trials

llm = ChatOllama(model="llama3", temperature=0)

# Instanciamos los parsers estructurados
structured_llm_eval = llm.with_structured_output(TrialEvaluation)
structured_llm_extract = llm.with_structured_output(PatientExtraction)

def retrieve_node(state: AgentState) -> AgentState:
    print("\n[Nodo 1] Analizando paciente y recuperando ensayos...")
    patient = state["patient_profile"]
    
    # 1. Extracción dinámica de la condición (Reemplaza el mock)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un experto médico. Extrae la condición principal del paciente para consultar la API de ClinicalTrials.gov. Limítate a extraer la condición principal en inglés y 3 términos MeSH."),
        ("user", "{patient}")
    ])
    extraction = (prompt | structured_llm_extract).invoke({"patient": patient})
    print(f"   -> Condición detectada: {extraction.primary_condition}")
    
    # 2. Llamada real a la API
    trials = fetch_clinical_trials(extraction.primary_condition, max_results=10)
    
    return {"retrieved_trials": trials, "mesh_terms": extraction.mesh_terms}

def evaluate_node(state: AgentState) -> AgentState:
    print("\n[Nodo 2] Verificación de elegibilidad estricta...")
    patient = state["patient_profile"]
    trials = state["retrieved_trials"]
    evaluated_list = []

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un experto oncólogo e investigador clínico. Compara el perfil del paciente con los criterios.
        1. Evalúa CADA criterio: "met", "not met", o "not enough info".
        2. Si marcas "not enough info", genera la 'missing_clinical_question' exacta.
        3. 'is_eligible' DEBE ser False si existe al menos un criterio 'not met'."""),
        ("user", "PACIENTE:\n{patient}\n\nCRITERIOS:\n{criteria_text}\n\nID ENSAYO: {nct_id}")
    ])
    chain = prompt | structured_llm_eval

    for study in trials:
        nct_id = study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
        criteria_text = study.get("protocolSection", {}).get("eligibilityModule", {}).get("eligibilityCriteria", "")

        try:
            result = chain.invoke({
                "patient": patient,
                "criteria_text": criteria_text,
                "nct_id": nct_id
            })
            
            # Lógica dura de elegibilidad y Scoring
            total = len(result.criteria)
            met_count = sum(1 for c in result.criteria if c.status == 'met')
            nei_count = sum(1 for c in result.criteria if c.status == 'not enough info')
            not_met_count = sum(1 for c in result.criteria if c.status == 'not met')
            
            result.is_eligible = (not_met_count == 0)
            result.score = (met_count + (0.5 * nei_count)) / total if result.is_eligible and total > 0 else 0.0
            
            evaluated_list.append(result)
            print(f"   - {nct_id} | Eligible: {result.is_eligible} | Score: {result.score:.2f}")
        except Exception as e:
            print(f"   - Error con {nct_id}: {e}")

    evaluated_list.sort(key=lambda x: x.score, reverse=True)
    return {"evaluated_trials": evaluated_list}

def format_dossier_node(state: AgentState) -> AgentState:
    # Este nodo ya no guarda a disco, solo limpia el estado para el batch
    print("\n[Nodo 3] Formateando Dosier...")
    return state # El output final se gestiona en el script de evaluación

# CONSTRUCCIÓN DEL GRAFO (Una sola vez)
workflow = StateGraph(AgentState)
workflow.add_node("retrieve_trials", retrieve_node)
workflow.add_node("evaluate_eligibility", evaluate_node)
workflow.add_node("format_dossier", format_dossier_node)

workflow.set_entry_point("retrieve_trials")
workflow.add_edge("retrieve_trials", "evaluate_eligibility")
workflow.add_edge("evaluate_eligibility", "format_dossier")
workflow.add_edge("format_dossier", END)

app = workflow.compile()
