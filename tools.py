import requests
from typing import List, Dict

def fetch_clinical_trials(condition: str, max_results: int = 10) -> List[Dict]:
    """Llama a la API REST de ClinicalTrials.gov."""
    # Usamos la API v2 (el estándar actual derivado del data-api)
    url = "https://clinicaltrials.gov/api/v2/studies"
    params = {
        "query.cond": condition,
        "filter.overallStatus": "RECRUITING",
        "pageSize": max_results
    }
    
    try:
        print(f"-> Buscando en API ensayos para: {condition}...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        studies = data.get("studies", [])
        print(f"-> ¡{len(studies)} ensayos recuperados!")
        return studies
    except Exception as e:
        print(f"-> Error consultando ClinicalTrials.gov: {e}")
        return []
