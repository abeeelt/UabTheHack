import requests
import xml.etree.ElementTree as ET
from typing import List, Dict

def fetch_clinical_trials(condition: str, max_results: int = 5) -> List[Dict]:
    """Descarga ensayos, pero SOLO los campos necesarios para no saturar la RAM."""
    url = "https://clinicaltrials.gov/api/v2/studies"
    params = {
        "query.cond": condition,
        "filter.overallStatus": "RECRUITING",
        "pageSize": max_results,
        "fields": "NCTId,OfficialTitle,EligibilityModule" # CRÍTICO: Previene el colapso del LLM
    }
    try:
        print(f"-> [API] Buscando ensayos para: {condition}...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get("studies", [])
    except Exception as e:
        print(f"-> Error en API: {e}")
        return []

def parse_topics(xml_path: str) -> List[Dict]:
    """Lee los pacientes del archivo XML de TREC."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        topics = []
        for topic in root.findall("topic"):
            topic_id = int(topic.get("number"))
            template = topic.get("template")
            fields = {field.get("name"): (field.text or "").strip() for field in topic.findall("field") if (field.text or "").strip()}
            
            lines = [f"Condition: {template}"] + [f"  {name}: {value}" for name, value in fields.items()]
            topics.append({"id": topic_id, "profile_text": "\n".join(lines)})
        return topics
    except Exception as e:
        print(f"Error leyendo {xml_path}: {e}")
        return []