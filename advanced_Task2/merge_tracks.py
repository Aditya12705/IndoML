import argparse
import json
import os
from typing import Dict, List


def read_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mi_path", required=True)
    p.add_argument("--pg_path", required=True)
    p.add_argument("--output_path", required=True)
    args = p.parse_args()

    mi = {x["conversation_id"]: x for x in read_json(args.mi_path)}
    pg = {x["conversation_id"]: x for x in read_json(args.pg_path)}

    merged: List[Dict] = []
    for cid, obj in mi.items():
        base = {
            "conversation_id": cid,
            "conversation_history": obj.get("conversation_history", ""),
            "tutor_responses": {},
        }
        # Merge tutors
        tutors = set(list(obj.get("tutor_responses", {}).keys())) | set(list(pg[cid].get("tutor_responses", {}).keys()))
        for t in tutors:
            base["tutor_responses"][t] = {
                "response": obj.get("tutor_responses", {}).get(t, pg[cid].get("tutor_responses", {}).get(t, {})).get("response", ""),
                "annotation": {},
            }
            if t in obj.get("tutor_responses", {}):
                ann = obj["tutor_responses"][t].get("annotation", {})
                if "Mistake_Identification" in ann:
                    base["tutor_responses"][t]["annotation"]["Mistake_Identification"] = ann["Mistake_Identification"]
            if t in pg[cid].get("tutor_responses", {}):
                ann = pg[cid]["tutor_responses"][t].get("annotation", {})
                if "Providing_Guidance" in ann:
                    base["tutor_responses"][t]["annotation"]["Providing_Guidance"] = ann["Providing_Guidance"]

        merged.append(base)

    write_json(args.output_path, merged)


if __name__ == "__main__":
    main()


