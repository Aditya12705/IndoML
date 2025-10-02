def collate_fn(batch):
    # Return batch as a list of dicts, not dict of lists
    return batch
import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from .modeling_singlehead import SingleHeadConfig, SingleHeadClassifier


def read_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class InferenceDataset(Dataset):
    def __init__(self, items: List[Dict], tokenizer, max_len: int = 512) -> None:
        self.samples: List[Dict] = []
        self.tok = tokenizer
        self.max_len = max_len
        for item in items:
            conv = item.get("conversation_history", "")
            for name, resp in item.get("tutor_responses", {}).items():
                self.samples.append({
                    "conversation_id": item.get("conversation_id"),
                    "conversation_history": conv,
                    "tutor": name,
                    "response": resp.get("response", ""),
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        # Ensure non-empty strings for tokenizer
        ctx = s["conversation_history"]
        rsp = s["response"]
        # If batched as list, use first element
        if isinstance(ctx, list):
            ctx = ctx[0] if ctx else " "
        if isinstance(rsp, list):
            rsp = rsp[0] if rsp else " "
        ctx = ctx if ctx else " "
        rsp = rsp if rsp else " "
        return {"conversation_history": ctx, "response": rsp, "meta": s}


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--track", choices=["mi", "pg"], required=True)
    p.add_argument("--model_name", default="microsoft/deberta-v3-base")
    p.add_argument("--models_dir", default=os.path.join("IndoML_Datathon", "advanced_Task2", "models"))
    p.add_argument("--input_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()

    label_idx2str = {0: "No", 1: "To some extent", 2: "Yes"}
    key = "Mistake_Identification" if args.track == "mi" else "Providing_Guidance"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    items = read_json(args.input_path)
    ds = InferenceDataset(items, tok, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    ckpts = sorted([p for p in os.listdir(args.models_dir) if p.startswith(f"{args.track}_fold_") and p.endswith(".pt")])
    cfg = SingleHeadConfig(backbone_name=args.model_name)
    models = []
    for pth in ckpts:
        ckpt = torch.load(os.path.join(args.models_dir, pth), map_location="cpu")
        m = SingleHeadClassifier(cfg)
        m.load_state_dict(ckpt["model"], strict=False)
        m.to(device)
        m.eval()
        models.append(m)

    preds = {}
    with torch.no_grad():
        for batch in dl:
            if not batch:
                continue  # skip empty batches
            # Prepare batch for tokenization
            ctxs = [(str(b["conversation_history"]) if b["conversation_history"] else " ") for b in batch]
            rsps = [(str(b["response"]) if b["response"] else " ") for b in batch]
            metas = [b["meta"] for b in batch]
            enc = tok(
                ctxs,
                rsps,
                max_length=args.max_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            batch_inputs = {k: v.to(device) for k, v in enc.items()}
            logits_sum = None
            for m in models:
                out = m(
                    input_ids=batch_inputs["input_ids"],
                    attention_mask=batch_inputs["attention_mask"],
                    token_type_ids=batch_inputs.get("token_type_ids"),
                )
                lg = out["logits"].detach().cpu().numpy()
                logits_sum = lg if logits_sum is None else logits_sum + lg
            prob = softmax(logits_sum, axis=1)
            for i in range(prob.shape[0]):
                meta = metas[i]
                conv_id = meta["conversation_id"]
                tutor = meta["tutor"]
                if isinstance(conv_id, list):
                    conv_id = conv_id[0] if conv_id else "unknown"
                if isinstance(tutor, list):
                    tutor = tutor[0] if tutor else "unknown"
                if conv_id not in preds:
                    preds[conv_id] = {
                        "conversation_id": conv_id,
                        "conversation_history": meta["conversation_history"],
                        "tutor_responses": {},
                    }
                if tutor not in preds[conv_id]["tutor_responses"]:
                    preds[conv_id]["tutor_responses"][tutor] = {"response": meta["response"], "annotation": {}}
                preds[conv_id]["tutor_responses"][tutor]["annotation"][key] = label_idx2str[int(np.argmax(prob[i]))]

    out_items: List[Dict] = []
    # keep input ordering
    missing_ids = []
    for item in items:
        cid = item.get("conversation_id")
        if cid in preds:
            out_items.append(preds[cid])
        else:
            missing_ids.append(cid)
    if missing_ids:
        print(f"[WARNING] Missing predictions for conversation_ids: {missing_ids}")
    write_json(args.output_path, out_items)


if __name__ == "__main__":
    main()


