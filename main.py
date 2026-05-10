from agent import app
from parse_topics import parse_topics


def main():
    topics = parse_topics("topics2023.xml")
    print(f"Cargados {len(topics)} perfiles de pacientes.\n")

    for topic in topics:
        print(f"{'='*60}")
        print(f"Topic {topic['id']} - {topic['template'].upper()}")
        print(f"{'='*60}")
        print(topic["profile_text"])
        print()

        initial_state = {
            "patient_profile": topic["profile_text"],
            "mesh_terms": [],
            "retrieved_trials": [],
            "evaluated_trials": [],
        }

        final_state = app.invoke(initial_state)

        evaluated = final_state.get("evaluated_trials", [])
        print(f"\nResultados Topic {topic['id']} ({len(evaluated)} ensayos evaluados):")
        for trial in evaluated:
            nct_id = trial.nct_id if hasattr(trial, "nct_id") else trial.get("nct_id")
            score = trial.score if hasattr(trial, "score") else trial.get("score")
            eligible = trial.is_eligible if hasattr(trial, "is_eligible") else trial.get("is_eligible")
            print(f"  {nct_id} | Eligible: {eligible} | Score: {score:.2f}")
        print()


if __name__ == "__main__":
    main()
