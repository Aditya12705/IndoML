import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from .modeling_multitask import MultiTaskConfig, MultiTaskClassifier


def read_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_text(conversation_history: str, response: str) -> str:
    return f"[CONTEXT]\n{conversation_history}\n[RESPONSE]\n{response}"


class InferenceDataset(Dataset):
    def __init__(self, items: List[Dict], tokenizer, max_len: int = 512) -> None:
        self.samples: List[Dict] = []
        self.tok = tokenizer
        self.max_len = max_len
        for item in items:
            conv = item.get("conversation_history", "")
            for name, resp in item.get("tutor_responses", {}).items():
                text = build_text(conv, resp.get("response", ""))
                self.samples.append({
                    "conversation_id": item.get("conversation_id"),
                    "conversation_history": conv,
                    "tutor": name,
                    "response": resp.get("response", ""),
                })
                self.samples[-1]["text"] = text

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        enc = self.tok(
            s["text"],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["meta"] = {
            "conversation_id": s["conversation_id"],
            "conversation_history": s["conversation_history"],
            "tutor": s["tutor"],
            "response": s["response"],
        }
        return item


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="microsoft/deberta-v3-base")
    parser.add_argument("--models_dir", default=os.path.join("IndoML_Datathon", "advanced", "models"))
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    items = read_json(args.input_path)
    ds = InferenceDataset(items, tokenizer, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Load fold checkpoints
    ckpts = sorted([p for p in os.listdir(args.models_dir) if p.startswith("fold_") and p.endswith(".pt")])
    models = []
    cfg = MultiTaskConfig(backbone_name=args.model_name)
    for p in ckpts:
        ckpt = torch.load(os.path.join(args.models_dir, p), map_location="cpu")
        m = MultiTaskClassifier(cfg)
        m.load_state_dict(ckpt["model"], strict=False)
        m.to(device)
        m.eval()
        models.append(m)

    label_map = {0: "No", 1: "To some extent", 2: "Yes"}

    # Collect predictions
    preds: Dict[str, Dict] = {}
    with torch.no_grad():
        for batch in dl:
            metas = batch.pop("meta")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits_mi = None
            logits_pg = None
            for m in models:
                out = m(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids"),
                )
                lm = out["logits_mi"].detach().cpu().numpy()
                lp = out["logits_pg"].detach().cpu().numpy()
                logits_mi = lm if logits_mi is None else logits_mi + lm
                logits_pg = lp if logits_pg is None else logits_pg + lp

            # Soft voting
            prob_mi = softmax(logits_mi, axis=1)
            prob_pg = softmax(logits_pg, axis=1)

            idx = 0
            for i in range(prob_mi.shape[0]):
                meta = metas["conversation_id"][i], metas["conversation_history"][i], metas["tutor"][i], metas["response"][i]
                conv_id, conv_hist, tutor, response = meta
                mi_label = label_map[int(np.argmax(prob_mi[i]))]
                pg_label = label_map[int(np.argmax(prob_pg[i]))]
                key = conv_id
                if key not in preds:
                    preds[key] = {
                        "conversation_id": conv_id,
                        "conversation_history": conv_hist,
                        "tutor_responses": {},
                    }
                preds[key]["tutor_responses"][tutor] = {
                    "response": response,
                    "annotation": {
                        "Mistake_Identification": mi_label,
                        "Providing_Guidance": pg_label,
                    },
                }
                idx += 1

    # Order according to input
    out_items: List[Dict] = []
    for item in items:
        out_items.append(preds[item.get("conversation_id")])

    write_json(args.output_path, out_items)


if __name__ == "__main__":
    main()


