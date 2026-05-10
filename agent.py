from langgraph.graph import StateGraph, END
from models import AgentState

# Funciones dummy (nodos)
def extract_terms(state: AgentState):
    print("-> Extrayendo términos MeSH...")
    return {"mesh_terms": ["Término 1"]}

def retrieve_trials(state: AgentState):
    print("-> Buscando en ClinicalTrials.gov...")
    return {"retrieved_trials": [{"nctId": "NCT123"}]}

def evaluate(state: AgentState):
    print("-> Evaluando criterios médicos...")
    return {"evaluated_trials": []}

def rank(state: AgentState):
    print("-> Generando ranking razonado...")
    return state

# Compilación del grafo
workflow = StateGraph(AgentState)

workflow.add_node("extract", extract_terms)
workflow.add_node("retrieve", retrieve_trials)
workflow.add_node("evaluate", evaluate)
workflow.add_node("rank", rank)

workflow.set_entry_point("extract")
workflow.add_edge("extract", "retrieve")
workflow.add_edge("retrieve", "evaluate")
workflow.add_edge("evaluate", "rank")
workflow.add_edge("rank", END)

app = workflow.compile()
