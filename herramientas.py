#Grupo formado por: Abel, Guilherme, Santi y Uven


#··························
#EXTRAER DATOS
#··························

import requests
import xml.etree.ElementTree as ET
from typing import List, Dict


def fetch_clinical_trials(condition: str, max_results: int = 5) -> List[Dict]:  #Esto para conectar con la api de ClinicalTrials
    """Descarga ensayos, pero SOLO los campos necesarios para no saturar la RAM."""
    url = "https://clinicaltrials.gov/api/v2/studies"
    params = {
        "query.cond": condition, #Filtramos por la enfermadad de PatientExtraction.primary_condition
        "filter.overallStatus": "RECRUITING", #que esten reclutando gente
        "pageSize": max_results,
        "fields": "NCTId,OfficialTitle,EligibilityModule" # Para simplificar la respuesta de la api y que no mande un monton de informacion, solo titulo y criterios
    }
    try:
        print(f"-> [API] Buscando ensayos para: {condition}...")
        response = requests.get(url, params=params)  # petición HTTP GET sin autenticación,
        response.raise_for_status()
        return response.json().get("studies", [])
    except Exception as e:
        print(f"-> Error en API: {e}") #falla bastantes veces
        return []

def parse_topics(xml_path: str) -> List[Dict]: #Para leer los pacientes del TREC
    """Lee los pacientes del archivo XML de TREC."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        topics = []
        for topic in root.findall("topic"):
            topic_id = int(topic.get("number")) #ID del paciente
            template = topic.get("template") #Condicion general
            fields = {field.get("name"): (field.text or "").strip() for field in topic.findall("field") if (field.text or "").strip()} #Mas informacion
            
            #Con esto cogemos toda la informacion y la ponemos en un parrafo profile_text --> patient_profile
            lines = [f"Condition: {template}"] + [f"  {name}: {value}" for name, value in fields.items()]
            topics.append({"id": topic_id, "profile_text": "\n".join(lines)})
        return topics
    except Exception as e:
        print(f"Error leyendo {xml_path}: {e}")
        return []
