# Agente de Cribratge Clínic — Deloitte × UAB Hackathon

**Equipo:** Abel, Guilherme Jimenez Aguas, Santi y Uven

Sistema de agente inteligente que automatiza el cribratge de pacientes en ensayos clínicos. Dado el perfil de un paciente en lenguaje natural, el agente busca ensayos relevantes en ClinicalTrials.gov y evalúa la elegibilidad del paciente criterio por criterio.

---

## Arquitectura

El sistema se implementa como un **grafo de estados con LangGraph**.

## Estructura del proyecto
HeCocinado/
├── main.py # Grafo LangGraph, nodos y bucle principal
├── models.py # Esquemas Pydantic (AgentState, TrialEvaluation, StructuredPatient...)
└── herramientas.py # Conexión con la API de ClinicalTrials.gov y parser XML de TREC



---

## Requisitos

```bash
pip install langgraph langchain-openai pydantic requests
```
El modelo de lenguaje se sirve en local mediante LM Studio en http://localhost:1234/v1. Cualquier modelo compatible con la API de OpenAI funciona; se recomienda uno de al menos 8B parámetros.

Uso de:
python main.py

El script lee los perfiles de pacientes de topics2023.xml, procesa cada uno a través del grafo y genera:

results_mejorado.json — ranking de ensayos por paciente con scores y detalle de criterios
dosier_paciente_<id>.md — dossier clínico en Markdown por cada paciente (Tarea 5)
El sistema incluye checkpointing: si la ejecución se interrumpe, al relanzar detecta el progreso guardado en el JSON y continúa desde el último paciente procesado.

