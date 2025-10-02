import argparse
import json
import os
from typing import Dict, List, Tuple

from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder


def read_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_text(conversation_history: str, response: str) -> str:
    # Concatenate conversation and response for context-aware classification
    return f"[CONTEXT]\n{conversation_history}\n[RESPONSE]\n{response}"


def extract_training_examples(train_items: List[Dict]) -> Tuple[List[str], List[str], List[str]]:
    texts: List[str] = []
    labels_mi: List[str] = []
    labels_pg: List[str] = []
    for item in train_items:
        conv = item.get("conversation_history", "")
        tutor_responses = item.get("tutor_responses", {})
        for _name, resp_obj in tutor_responses.items():
            resp = resp_obj.get("response", "")
            ann = resp_obj.get("annotation", {})
            mi = ann.get("Mistake_Identification")
            pg = ann.get("Providing_Guidance")
            if mi is None or pg is None:
                # Skip malformed entries
                continue
            texts.append(build_text(conv, resp))
            labels_mi.append(mi)
            labels_pg.append(pg)
    return texts, labels_mi, labels_pg


def create_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                    max_features=150000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            ("clf", LinearSVC()),
        ]
    )


def train_models(train_path: str, model_dir: str) -> None:
    os.makedirs(model_dir, exist_ok=True)
    train_items = read_json(train_path)
    X_texts, y_mi, y_pg = extract_training_examples(train_items)

    if not X_texts:
        raise ValueError("No training examples were found. Check the training JSON format.")

    # Encoders for labels
    le_mi = LabelEncoder().fit(y_mi)
    le_pg = LabelEncoder().fit(y_pg)

    y_mi_enc = le_mi.transform(y_mi)
    y_pg_enc = le_pg.transform(y_pg)

    model_mi = create_pipeline()
    model_pg = create_pipeline()

    model_mi.fit(X_texts, y_mi_enc)
    model_pg.fit(X_texts, y_pg_enc)

    dump(model_mi, os.path.join(model_dir, "model_mi.joblib"))
    dump(model_pg, os.path.join(model_dir, "model_pg.joblib"))
    dump(le_mi, os.path.join(model_dir, "labels_mi.joblib"))
    dump(le_pg, os.path.join(model_dir, "labels_pg.joblib"))


def predict_for_split(input_path: str, output_path: str, model_dir: str, include_history: bool = True) -> None:
    items = read_json(input_path)
    model_mi = load(os.path.join(model_dir, "model_mi.joblib"))
    model_pg = load(os.path.join(model_dir, "model_pg.joblib"))
    le_mi: LabelEncoder = load(os.path.join(model_dir, "labels_mi.joblib"))
    le_pg: LabelEncoder = load(os.path.join(model_dir, "labels_pg.joblib"))

    out_items: List[Dict] = []
    for item in items:
        conv = item.get("conversation_history", "")
        tutor_responses = item.get("tutor_responses", {})
        new_tr: Dict[str, Dict] = {}
        for name, resp_obj in tutor_responses.items():
            resp = resp_obj.get("response", "")
            text = build_text(conv, resp) if include_history else resp

            mi_pred = model_mi.predict([text])[0]
            pg_pred = model_pg.predict([text])[0]
            mi_label = le_mi.inverse_transform([mi_pred])[0]
            pg_label = le_pg.inverse_transform([pg_pred])[0]

            new_tr[name] = {
                "response": resp,
                "annotation": {
                    "Mistake_Identification": mi_label,
                    "Providing_Guidance": pg_label,
                },
            }

        out_items.append(
            {
                "conversation_id": item.get("conversation_id"),
                "conversation_history": conv,
                "tutor_responses": new_tr,
            }
        )

    write_json(output_path, out_items)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline and generate predictions.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train", help="Train models on trainset.json")
    train_p.add_argument("--train_path", default=os.path.join("IndoML_Datathon", "data", "trainset.json"))
    train_p.add_argument("--model_dir", default=os.path.join("IndoML_Datathon", "baseline", "models"))

    pred_p = subparsers.add_parser("predict", help="Predict for a split and write predictions.json")
    pred_p.add_argument("--input_path", required=True)
    pred_p.add_argument("--output_path", required=True)
    pred_p.add_argument("--model_dir", default=os.path.join("IndoML_Datathon", "baseline", "models"))
    pred_p.add_argument("--no_history", action="store_true", help="Use only response text (ignore history)")

    args = parser.parse_args()

    if args.command == "train":
        train_models(args.train_path, args.model_dir)
    elif args.command == "predict":
        predict_for_split(
            input_path=args.input_path,
            output_path=args.output_path,
            model_dir=args.model_dir,
            include_history=not args.no_history,
        )
    else:
        raise ValueError("Unknown command")


if __name__ == "__main__":
    main()


