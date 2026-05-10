from agent import app
from models import AgentState

def main():
    initial_state = AgentState(patient_profile="Hombre de 65 años con diabetes tipo 2...")
    
    print("Iniciando ejecución del Agente...")
    final_state = app.invoke(initial_state.model_dump())
    
    print("\nEjecución completada.")

if __name__ == "__main__":
    main()
