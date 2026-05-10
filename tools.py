# api_tools.py
import requests
from typing import List, Dict

def fetch_clinical_trials(condition: str, max_results: int = 5) -> List[Dict]:
    url = "https://clinicaltrials.gov/api/v2/studies"
    params = {
        "query.cond": condition,
        "filter.overallStatus": "RECRUITING",
        "pageSize": max_results
    }
    try:
        print(f"-> [API] Buscando ensayos para: {condition}...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("studies", [])
    except Exception as e:
        print(f"-> Error en API: {e}")
        return []
