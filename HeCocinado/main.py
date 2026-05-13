#Grupo formado por: Abel, Guilherme, Santi y Uven


#··························
#MAIN
#··························

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os #Para leer archivos
import json

from models import AgentState, TrialEvaluation, PatientExtraction, StructuredPatient
from herramientas import fetch_clinical_trials, parse_topics

#CONFIGURAR EL MODELO --> LM Studio 
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="local-model",
    temperature=0, #Determinista, No creatiu

    max_retries=1, #Esto lo hemos añadido porque con el paciente 15 entra en un bucle en el Nodo1 al estructurar el paciente.
    timeout=150.0 #Si tarda demasiado no lo estructura y usa el texto crudo
)

structured_llm_extract = llm.with_structured_output(PatientExtraction) #Que el modelo  responda con los formatos de models.py
structured_llm_eval = llm.with_structured_output(TrialEvaluation)
structured_llm_patient = llm.with_structured_output(StructuredPatient)

#NODOS DEL GRAFO
#1. Estructurar Paciente
def structure_patient_node(state: AgentState) -> AgentState:
    print("\n[Nodo 1] Estructurando perfil del paciente...")
    patient_raw = state["patient_profile"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extrae la información del paciente en JSON estricto. Idioma: ESPAÑOL. Si falta un dato, déjalo vacío o nulo."),
        ("user", "{patient}")
    ])
    
    try:
        clean_patient = (prompt | structured_llm_patient).invoke({"patient": patient_raw})
        clean_patient_str = clean_patient.model_dump_json(indent=2)
        print("   -> Paciente estructurado con éxito.")
    except Exception as e: #Si el LLM falla o se queda en bucle, (hemos puesto un timeout en la configuracion) --> Usa el texto crudo sin estructurar
        print(f"   -> Error estructurando. Usando texto bruto.")
        clean_patient_str = patient_raw
        
    return {"structured_patient": clean_patient_str}


#2. Buscar y recuperar (Tarea 1)
def retrieve_node(state: AgentState) -> AgentState:
    print("\n[Nodo 2] Buscando ensayos clínicos...")
    prompt = ChatPromptTemplate.from_messages([ #Sacamos la condicion medica principal y pegarsela a la API
        ("system", "Extrae la condición médica principal en inglés para buscar en la base de datos."),
        ("user", "{patient}")
    ])
    extraction = (prompt | structured_llm_extract).invoke({"patient": state["patient_profile"]})
    
    condicion_principal = extraction.primary_condition
    
    #Primero intentamos Buscando específicamente con 1 termino MeSH (algo simple)
    terminos_limpios = extraction.mesh_terms[:1] if extraction.mesh_terms else []
    query_compleja = f"{condicion_principal} {' '.join(terminos_limpios)}".strip()
    
    print(f"   -> Intentando búsqueda específica: {query_compleja}")
    trials_finales = fetch_clinical_trials(query_compleja, max_results=3) 
    
    #Si no se han encontrado tres ensayos--> Completamos hasta los 3 pasando de buscar especificamente --> buscamos con la condicion general, NUNCA FALLA
    if len(trials_finales) < 3:
        faltantes = 3 - len(trials_finales)
        print(f"   -> [!] Solo se encontraron {len(trials_finales)} ensayos específicos.")
        print(f"   -> [Fallback] Rellenando con búsqueda general: {condicion_principal}")
        
        trials_generales = fetch_clinical_trials(condicion_principal, max_results=3)
        
        #Cogemos los IDs que ya tenemos para NO DUPLICAR ni meterle el mismo ensayo dos veces
        ids_existentes = set()
        for t in trials_finales:
            nctid = t.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
            if nctid: ids_existentes.add(nctid)
            
        # Añadimos los generales hasta llegar a 3
        for t in trials_generales:
            if len(trials_finales) >= 3:
                break # Ya tenemos los 3 ensayos
                
            nctid = t.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
            if nctid and nctid not in ids_existentes:
                trials_finales.append(t)
                ids_existentes.add(nctid)
                
    print(f"   -> Ensayos recuperados finales: {len(trials_finales)}")
    
    return {"retrieved_trials": trials_finales, "mesh_terms": extraction.mesh_terms}

#3. Verificar (Tarea 2 y 4) y Ranking (Tarea3)
def evaluate_node(state: AgentState) -> AgentState:
    print("\n[Nodo 3] Verificación de elegibilidad estricta con CoT...")
    clean_patient = state.get("structured_patient", "") 
    raw_patient = state["patient_profile"]
    trials = state["retrieved_trials"]
    evaluated_list = []

    #Prompt de la evaluación, se ha modificado muchisimas veces añadiendo reglas en base a errores que daba el LLM, es un modelo muy simple asi que le cuesta. 
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un evaluador clínico estricto y literal. Tu tarea es cruzar el perfil de un paciente con los criterios de un ensayo clínico.
        
        REGLAS ABSOLUTAS E INQUEBRANTABLES:
        1. IDIOMA: Piensa y responde ÚNICAMENTE en ESPAÑOL (excepto el texto en inglés del criterio).
        2. VERIFICACIÓN CRUCIAL (Chain of Thought): Antes de dar el estado, usa el campo 'verification_step' para analizar. Comprueba las afirmaciones (sí/no) en el TEXTO ORIGINAL DEL PACIENTE. Ejemplo: si dice "prior cataract surgery: no", significa que NO se ha operado.
        3. ESTADOS ESTRICTOS: El campo 'status' DEBE SER EXACTAMENTE uno de estos tres valores: "met", "not_met", o "not_enough_info".
        4. PREGUNTA OBLIGATORIA: Si el status es 'not_enough_info', genera la 'missing_clinical_question'. Si es 'met' o 'not_met', escribe estrictamente "N/A".
        5. REGLA ANTI-ALUCINACIONES MATIZADA: PARA LA MAYORIA DE LOS DATOS SOBRETODO CLÍNICOS (ej. edad, genero, capacidad mental, enfermedades, alergias, embarazos, cirugías previas, valores de laboratorio): La ausencia de información NO es cumplimiento. CERO SUPOSICIONES. Si el texto NO lo menciona explícitamente, el estado DEBE ser "not_enough_info". Unica y rara excepcion: si es un  dato estandar o administrativo (ej. capacidad o voluntad para firmar consentimiento informado, seguir instrucciones del estudio, asistir a citas, contestar cuestionarios): Aplica el "Principio de Presunción de Cumplimiento", si el perfil del paciente NO indica expresamente una incapacidad para hacerlo (ej. no menciona demencia severa, rechazo a participar o barreras logísticas), ASUME que el paciente es un adulto funcional capaz de cumplirlo. El estado DEBE ser "met".    
        6. LÓGICA DE INCLUSIÓN VS EXCLUSIÓN: El estado "met" significa que el paciente SUPERA el filtro de ese criterio. SI PUEDE HABER AUSENCIA DE INFORMACION CONSULTAR PASO 5
            - INCLUSIÓN (Debe tener X): Si lo tiene -> "met". Si no lo tiene -> "not_met".
            - EXCLUSIÓN (NO debe tener Y): Si LO TIENE -> "not_met" (es excluido del ensayo).
            - EXCLUSIÓN (NO debe tener Y): Si PONE EXPLICITAMENTE QUE NO LO TIENE -> "met" (pasa el filtro, es una condición favorable). En caso de ausencia de informacion not_enough_info

        EJEMPLO DE EXCLUSIÓN SUPERADA:
            {{
            "criterion_text": "History of cataract surgery within 6 months.",
            "verification_step": "Es un criterio de exclusión (prohíbe tener cirugía). El texto original del paciente indica 'prior cataract surgery: no'.",
            "status": "met",
            "reasoning": "Al no haberse operado de cataratas, el paciente supera satisfactoriamente este filtro de exclusión.",
            "missing_clinical_question": "N/A"
            }}
        7. PROHIBIDO TRADUCIR EL CRITERIO: El campo 'criterion_text' debe ser una copia exacta (copy-paste) del texto en inglés. No lo traduzcas al español bajo ninguna circunstancia.
        8. CORRELACIONES MÉDICAS INVENTADAS: Si el criterio pide "Enfermedad Macular" y el paciente tiene "Edema Corneal", no asumas que son lo mismo o que le afecta igual. Si no son médicamente equivalentes y no se menciona, es "not_enough_info" o "met" (si es exclusión y asumimos que no la tiene), pero NUNCA deduzcas una exclusión por una patología diferente.
        9. NO PREDECIR EL FUTURO: Si el criterio pide "Not expected to need glaucoma surgery in the next 2 months" y el paciente "no tiene cirugía previa", NUNCA deduzcas que no la necesitará en el futuro. Usa "not_enough_info".
        10. UMBRALES NUMÉRICOS ESTRICTOS: Si un criterio pide un valor numérico (ej. "visión peor de -3 dB") y el perfil solo da texto cualitativo (ej. "daño moderado"), el estado ES ESTRICTAMENTE "not_enough_info". No adivines equivalencias.
        EJEMPLO DE RAZONAMIENTO PERFECTO:
        {{
          "criterion_text": "History of cataract surgery within 6 months.",
          "verification_step": "Busco cirugía de cataratas. El texto original dice 'prior cataract surgery: no'. Por tanto, no se ha operado.",
          "status": "met",
          "reasoning": "El paciente no tiene historial de cirugía de cataratas, cumpliendo el requisito de exclusión.",
          "missing_clinical_question": "N/A"
        }}
        
        Devuelve ÚNICAMENTE un JSON válido."""),
        ("user", "TEXTO ORIGINAL DEL PACIENTE:\n{raw_patient}\n\nRESUMEN ESTRUCTURADO:\n{clean_patient}\n\nCRITERIOS DEL ENSAYO:\n{criteria_text}\n\nID ENSAYO: {nct_id}")
    ])
    chain = prompt | structured_llm_eval

    for study in trials: #Analizamos cada ensayo
        nct_id = study.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "N/A")
        criteria_text = study.get("protocolSection", {}).get("eligibilityModule", {}).get("eligibilityCriteria", "")
        
        #Añadimos esto a partir del PACIENTE 15 para que el modelo local solo evalue unos 3 o 4 Criterios y que vaya más rápido. Con todos los criterios esta dedicando unos 10 minutos por paciente que es demasiado.
        if len(criteria_text) > 1000:
            criteria_text = criteria_text[:1000] + "..."
        
        
        if not criteria_text:
            continue

        try:
            result = chain.invoke({
                "raw_patient": raw_patient,
                "clean_patient": clean_patient,
                "criteria_text": criteria_text,
                "nct_id": nct_id
            })
            
        #Calculamos en Python, antes le dejabamos al modelo calcular el ranking pero es mejor hacerlo de esta manera para no saturarlo y dar resultado que siempre sigan la misma logica
            total = len(result.criteria)
            met_count = sum(1 for c in result.criteria if c.status == 'met')
            nei_count = sum(1 for c in result.criteria if c.status == 'not_enough_info')
            not_met_count = sum(1 for c in result.criteria if c.status == 'not_met')
            

            if not_met_count > 0: #Con un solo not met ya no es elegible
                is_eligible = "No Elegible"
            elif met_count == 0 and nei_count > 0: #Hemos añadido el estado posible por si todos los criterios los pasa como not_enough_info
                is_eligible = "Posible"
            else:
                is_eligible = "Elegible"

            #Nuestro calculo del score
            score = (met_count + (0.5 * nei_count)) / total if total > 0 else 0.0 
            
            #gardamos un diccionario en el estado con el ID, el score y los resultados
            evaluated_list.append({
                "nct_id": result.nct_id,
                "is_eligible": is_eligible,
                "score": score,
                "criteria": result.criteria
            })
            print(f"   - {nct_id} | Eligible: {is_eligible} | Score: {score:.2f}")
        except Exception as e:
            print(f"   - Error procesando {nct_id}: {e}")

    #ordenamos la lista de diccionarios por score pero damos prioridad segun los Elegible, Posible y No elegible
    evaluated_list.sort(
        key=lambda x: (2 if x["is_eligible"] == "Elegible" else 1 if x["is_eligible"] == "Posible" else 0, x["score"]), 
        reverse=True
    )    
    return {"evaluated_trials": evaluated_list}


    
#Construimos el gafo (LangGraph) --> Añadimos los nodos y conectamos como flujo lineal
workflow = StateGraph(AgentState)
workflow.add_node("structure_patient", structure_patient_node) 
workflow.add_node("retrieve_trials", retrieve_node) 
workflow.add_node("evaluate_eligibility", evaluate_node) 

workflow.set_entry_point("structure_patient")
workflow.add_edge("structure_patient", "retrieve_trials")
workflow.add_edge("retrieve_trials", "evaluate_eligibility")
workflow.add_edge("evaluate_eligibility", END)
app = workflow.compile()

#Generamos un dosier para cada paciente con los resultados del json(Tarea 5)
def generar_dosier_markdown(topic_id, resultados_paciente):
    md_content = f"# Dosier de Preselección - Paciente {topic_id}\n\n"
    for trial in resultados_paciente:

        estado_str = trial.get("is_eligible", "Error")
        if estado_str == "Elegible":
            estado = "✅ ELEGIBLE"
        elif estado_str == "No Elegible":
            estado = "❌ NO ELEGIBLE"
        elif estado_str == "Posible":
            estado = "⚠️ POSIBLE (Falta Información)"
        else:
            estado = "🔴 ERROR DE EVALUACIÓN"
            
        md_content += f"## Ensayo: {trial['nct_id']} | Estado: {estado} | Score: {trial['score']:.2f}\n\n"
        md_content += "| Criterio | Estado | Razón | Pregunta Faltante |\n"
        md_content += "|---|---|---|---|\n"
        for det in trial["detalles"]:
            criterio_limpio = det['criterio'].replace('\n', ' ')
            razon_limpia = det['razon'].replace('\n', ' ')
            md_content += f"| {criterio_limpio} | **{det['estado']}** | {razon_limpia} | {det['pregunta_faltante']} |\n"
        md_content += "\n---\n"
    
    with open(f"dosier_paciente_{topic_id}.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"-> Dosier generado: dosier_paciente_{topic_id}.md")

    




#Bucle Principal
if __name__ == "__main__":
    topics = parse_topics("topics2023.xml")
    if not topics:
        print("No se encontraron pacientes.")
        exit()

    print(f"Cargados {len(topics)} perfiles de pacientes.\n")
    
    archivo_resultados = "results_mejorado.json"
    resultados_finales = {}

    # --- LÓGICA DE CHECKPOINT (RETOMAR PROGRESO) ---
    if os.path.exists(archivo_resultados):
        try:
            with open(archivo_resultados, "r", encoding="utf-8") as f:
                resultados_finales = json.load(f)
            print(f"[*] Progreso detectado: {len(resultados_finales)} pacientes ya procesados.")
        except Exception as e:
            print("[!] El archivo JSON existe pero está corrupto o vacío. Empezando de cero.")

    # OJO: He quitado el [:1] para que recorra todos los pacientes
    for topic in topics:
        topic_id_str = str(topic['id'])
        
        # Si el paciente ya está en el JSON, nos lo saltamos
        if topic_id_str in resultados_finales:
            print(f"[*] Saltando Paciente ID: {topic_id_str} (Ya evaluado)")
            continue

        print(f"\n{'='*60}")
        print(f"Procesando Paciente ID: {topic['id']}")
        print(f"{'='*60}")
        
        initial_state = {
            "patient_profile": topic["profile_text"],
            "structured_patient": "",
            "mesh_terms": [],
            "retrieved_trials": [],
            "evaluated_trials": [],
        }

        # Ejecutamos el agente
        final_state = app.invoke(initial_state)
        
        ranking_paciente = []
        for trial in final_state.get("evaluated_trials", []):
            ranking_paciente.append({
                "nct_id": trial["nct_id"], 
                "is_eligible": trial["is_eligible"],
                "score": trial["score"],
                "detalles": [
                    {
                        "criterio": c.criterion_text[:80] + "...",
                        "verificacion": c.verification_step,
                        "estado": c.status,
                        "razon": c.reasoning,
                        "pregunta_faltante": c.missing_clinical_question
                    } for c in trial["criteria"]
                ]
            })
            
        # Añadimos el paciente recién procesado a los resultados
        resultados_finales[topic_id_str] = ranking_paciente

        # --- GUARDADO INCREMENTAL ---
        # Guardamos a disco DESPUÉS DE CADA PACIENTE. Si detienes el PC, no pierdes nada.
        with open(archivo_resultados, "w", encoding="utf-8") as f:
            json.dump(resultados_finales, f, indent=4, ensure_ascii=False)
            print(f"[*] Datos del Paciente {topic_id_str} guardados a disco correctamente.")
            
        # --- GENERAR DOSIER (TAREA 5) ---
        # Ahora SÍ llamamos a tu función para que cree el archivo Markdown
        generar_dosier_markdown(topic['id'], ranking_paciente)
        
    print("\n¡PROCESO COMPLETADO! Todos los pacientes han sido evaluados.")


