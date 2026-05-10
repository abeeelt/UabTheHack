# agent.py
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from models import AgentState, TrialEvaluation
from tools import fetch_clinical_trials

# 1. INICIALIZAMOS EL LLM LOCAL
llm = ChatOllama(model="llama3", temperature=0)
# Importante: Usamos structured_output para forzar el JSON exacto de Pydantic
structured_llm = llm.with_structured_output(TrialEvaluation)

# 2. DEFINIMOS EL NODO DE RECUPERACIÓN (Tarea 1)
def retrieve_node(state: AgentState) -> AgentState:
    print("\n[Nodo 1] Recuperando ensayos clínicos...")
    # En un sistema real extraeríamos la condición del patient_profile con otro LLM (MeSH). 
    # Por ahora, mockeamos la condición para avanzar rápido.
    trials = fetch_clinical_trials("Diabetes Type 2", max_results=5)
    
    # LangGraph actualiza el estado devolviendo un diccionario con la clave a actualizar
    return {"retrieved_trials": trials}

# 3. DEFINIMOS EL NODO DE EVALUACIÓN (Tarea 2 y 4)
# (Este es el código complejo de razonamiento médico)
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
    chain = prompt | structured_llm

    for study in trials:
        nct_id = study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
        criteria_text = study.get("protocolSection", {}).get("eligibilityModule", {}).get("eligibilityCriteria", "")

        try:
            result: TrialEvaluation = chain.invoke({
                "patient": patient,
                "criteria_text": criteria_text,
                "nct_id": nct_id
            })
            
            # Lógica dura de elegibilidad (Ranking - Tarea 3)
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
import json

def generate_dossier_node(state: AgentState) -> AgentState:
    print("\n[Nodo 3] Generando Dosier Estructurado (JSON)...")
    evaluated = state["evaluated_trials"]
    
    # Filtramos y ordenamos: solo elegibles o con score > 0, ordenados por score
    ranked_trials = [t.model_dump() for t in evaluated if t.score > 0]
    
    dossier = {
        "patient_profile": state["patient_profile"],
        "extracted_mesh": state["mesh_terms"],
        "recommended_trials": ranked_trials
    }
    
    # Guardar en disco (Requisito de entregable no interactivo)
    with open("predicciones_trec.json", "w", encoding="utf-8") as f:
        json.dump(dossier, f, indent=4, ensure_ascii=False)
        
    print(f"   -> ¡Dosier guardado con {len(ranked_trials)} ensayos recomendados!")
    return state # El estado no muta aquí

# --- EN LA CONSTRUCCIÓN DEL GRAFO ---
workflow.add_node("generate_dossier", generate_dossier_node)

workflow.set_entry_point("retrieve_trials")
workflow.add_edge("retrieve_trials", "evaluate_eligibility")
workflow.add_edge("evaluate_eligibility", "generate_dossier") # Conectamos al nuevo nodo
workflow.add_edge("generate_dossier", END)
# 4. CONSTRUIMOS EL GRAFO DE LANGGRAPH (El "Flujo")
workflow = StateGraph(AgentState)

# Añadimos los nodos al grafo
workflow.add_node("retrieve_trials", retrieve_node)
workflow.add_node("evaluate_eligibility", evaluate_node)

# Definimos el flujo lógico (Las flechas del diagrama)
workflow.set_entry_point("retrieve_trials")
workflow.add_edge("retrieve_trials", "evaluate_eligibility")
workflow.add_edge("evaluate_eligibility", END) # Por ahora termina aquí

# Compilamos el agente
app = workflow.compile()

# 5. EJECUCIÓN PRINCIPAL
if __name__ == "__main__":
    print("Iniciando Sistema Agéntico UAB THE HACK!...")
    
    # Estado inicial (El input del paciente)
    initial_state = {
        "patient_profile": "Hombre de 65 años con diabetes tipo 2...",
        "mesh_terms": [],
        "retrieved_trials": [],
        "evaluated_trials": []
    }
    
    # Ejecutamos el grafo completo
    final_state = app.invoke(initial_state)
    print("\nEjecución completada. Ensayos evaluados y ordenados listos para Dosier.")
