import xml.etree.ElementTree as ET
from typing import List, Dict


def parse_topics(xml_path: str = "topics2023.xml") -> List[Dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    topics = []
    for topic in root.findall("topic"):
        topic_id = int(topic.get("number"))
        template = topic.get("template")

        fields = {}
        for field in topic.findall("field"):
            name = field.get("name")
            value = (field.text or "").strip()
            if value:
                fields[name] = value

        profile_text = _build_profile_text(template, fields)
        topics.append({
            "id": topic_id,
            "template": template,
            "fields": fields,
            "profile_text": profile_text,
        })

    return topics


def _build_profile_text(template: str, fields: Dict[str, str]) -> str:
    lines = [f"Condition: {template}"]
    for name, value in fields.items():
        lines.append(f"  {name}: {value}")
    return "\n".join(lines)
