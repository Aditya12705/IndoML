import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .modeling_multitask import MultiTaskConfig, MultiTaskClassifier
from sklearn.metrics import f1_score, accuracy_score


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_text(conversation_history: str, response: str) -> str:
    return f"[CONTEXT]\n{conversation_history}\n[RESPONSE]\n{response}"


def read_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_examples(train_items: List[Dict]) -> Tuple[List[str], List[int], List[int]]:
    texts: List[str] = []
    y_mi: List[int] = []
    y_pg: List[int] = []
    label_map = {"No": 0, "To some extent": 1, "Yes": 2}
    for item in train_items:
        conv = item.get("conversation_history", "")
        for _name, resp in item.get("tutor_responses", {}).items():
            ann = resp.get("annotation", {})
            mi = ann.get("Mistake_Identification")
            pg = ann.get("Providing_Guidance")
            if mi is None or pg is None:
                continue
            text = build_text(conv, resp.get("response", ""))
            texts.append(text)
            y_mi.append(label_map[mi])
            y_pg.append(label_map[pg])
    return texts, y_mi, y_pg


class DialogDataset(Dataset):
    def __init__(self, contexts: List[str], responses: List[str], y_mi: List[int], y_pg: List[int], tokenizer, max_len: int = 512):
        self.contexts = contexts
        self.responses = responses
        self.y_mi = y_mi
        self.y_pg = y_pg
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ctx = self.contexts[idx]
        rsp = self.responses[idx]
        # Encode as pair; truncate only the first (context) so response is preserved
        enc = self.tok(
            ctx,
            rsp,
            max_length=self.max_len,
            truncation="only_first",
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels_mi"] = torch.tensor(self.y_mi[idx], dtype=torch.long)
        item["labels_pg"] = torch.tensor(self.y_pg[idx], dtype=torch.long)
        return item


@dataclass
class Args:
    model_name: str
    train_path: str
    out_dir: str
    folds: int
    epochs: int
    batch_size: int
    lr: float
    seed: int
    max_len: int
    grad_accum: int


def compute_class_weights(labels: List[int], num_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    inv = 1.0 / np.clip(counts, a_min=1.0, a_max=None)
    weights = inv / inv.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)

    def apply_shadow(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            backup[name] = param.detach().clone()
            param.data.copy_(self.shadow[name])
        return backup

    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])


class FGM:
    def __init__(self, model: nn.Module, epsilon: float = 1e-5) -> None:
        self.model = model
        self.epsilon = epsilon
        self.backup: Dict[nn.Parameter, torch.Tensor] = {}

    def attack(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and "embeddings" in name:
                self.backup[param] = param.data.clone()
                grad = param.grad
                norm = torch.norm(grad)
                if torch.isfinite(norm) and norm > 0:
                    r_at = self.epsilon * grad / norm
                    param.data.add_(r_at)

    def restore(self) -> None:
        for param, data in self.backup.items():
            param.data.copy_(data)
        self.backup.clear()


def evaluate(model: nn.Module, dl: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    all_mi, all_pg = [], []
    all_mi_pred, all_pg_pred = [], []
    loss_total = 0.0
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                labels_mi=batch["labels_mi"],
                labels_pg=batch["labels_pg"],
            )
            loss_total += out["loss"].item()
            mi_pred = out["logits_mi"].argmax(dim=1).detach().cpu().numpy()
            pg_pred = out["logits_pg"].argmax(dim=1).detach().cpu().numpy()
            all_mi_pred.extend(mi_pred)
            all_pg_pred.extend(pg_pred)
            all_mi.extend(batch["labels_mi"].detach().cpu().numpy())
            all_pg.extend(batch["labels_pg"].detach().cpu().numpy())
    loss_avg = loss_total / max(1, len(dl))
    f1_mi = f1_score(all_mi, all_mi_pred, average="macro")
    f1_pg = f1_score(all_pg, all_pg_pred, average="macro")
    f1_avg = 0.5 * (f1_mi + f1_pg)
    acc_mi = accuracy_score(all_mi, all_mi_pred)
    acc_pg = accuracy_score(all_pg, all_pg_pred)
    acc_avg = 0.5 * (acc_mi + acc_pg)
    return f1_avg, acc_avg, loss_avg


def train_fold(args: Args, fold_idx: int, train_idx: np.ndarray, val_idx: np.ndarray, contexts, responses, y_mi, y_pg, device: torch.device) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    ds_train = DialogDataset([contexts[i] for i in train_idx], [responses[i] for i in train_idx], [y_mi[i] for i in train_idx], [y_pg[i] for i in train_idx], tokenizer, args.max_len)
    ds_val = DialogDataset([contexts[i] for i in val_idx], [responses[i] for i in val_idx], [y_mi[i] for i in val_idx], [y_pg[i] for i in val_idx], tokenizer, args.max_len)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    cfg = MultiTaskConfig(backbone_name=args.model_name)
    model = MultiTaskClassifier(cfg).to(device)

    w_mi = compute_class_weights([y_mi[i] for i in train_idx]).to(device)
    w_pg = compute_class_weights([y_pg[i] for i in train_idx]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_training_steps = math.ceil(len(dl_train) * args.epochs / max(1, args.grad_accum))
    num_warmup = int(0.06 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup, num_training_steps=num_training_steps)

    best_score = -1.0
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"fold_{fold_idx}.pt")

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    ema = EMA(model, decay=0.997)
    fgm = FGM(model, epsilon=1e-5)
    patience = 2
    patience_left = patience

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step = 0
        for batch in dl_train:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids"),
                    labels_mi=batch["labels_mi"],
                    labels_pg=batch["labels_pg"],
                    loss_weights=(w_mi, w_pg),
                )
                loss = out["loss"] / max(1, args.grad_accum)
            scaler.scale(loss).backward()

            # adversarial step
            fgm.attack()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out_adv = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids"),
                    labels_mi=batch["labels_mi"],
                    labels_pg=batch["labels_pg"],
                    loss_weights=(w_mi, w_pg),
                )
                loss_adv = out_adv["loss"] / max(1, args.grad_accum)
            scaler.scale(loss_adv).backward()
            fgm.restore()

            step += 1
            if step % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)

        # Validation with EMA weights
        bak = ema.apply_shadow(model)
        f1_avg, acc_avg, val_loss = evaluate(model, dl_val, device)
        ema.restore(model, bak)

        score = 0.7 * f1_avg + 0.3 * (1.0 - min(1.0, val_loss))
        if score > best_score:
            best_score = score
            patience_left = patience
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, ckpt_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="microsoft/deberta-v3-base")
    parser.add_argument("--train_path", default=os.path.join("IndoML_Datathon", "data", "trainset.json"))
    parser.add_argument("--out_dir", default=os.path.join("IndoML_Datathon", "advanced", "models"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--grad_accum", type=int, default=1)
    args_ns = parser.parse_args()

    args = Args(
        model_name=args_ns.model_name,
        train_path=args_ns.train_path,
        out_dir=args_ns.out_dir,
        folds=args_ns.folds,
        epochs=args_ns.epochs,
        batch_size=args_ns.batch_size,
        lr=args_ns.lr,
        seed=args_ns.seed,
        max_len=args_ns.max_len,
        grad_accum=args_ns.grad_accum,
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = read_json(args.train_path)
    texts, y_mi, y_pg = extract_examples(data)
    # split texts into contexts and responses for pair encoding
    contexts: List[str] = []
    responses: List[str] = []
    for t in texts:
        # They were built with markers; split to context/response
        # [CONTEXT]\n...\n[RESPONSE]\n...
        parts = t.split("[RESPONSE]\n")
        ctx = parts[0].replace("[CONTEXT]\n", "") if len(parts) >= 1 else t
        rsp = parts[1] if len(parts) > 1 else ""
        contexts.append(ctx)
        responses.append(rsp)

    # Stratification by joint label pair to preserve distribution
    joint = [mi * 3 + pg for mi, pg in zip(y_mi, y_pg)]
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    for fold, (tr, va) in enumerate(skf.split(np.arange(len(texts)), joint)):
        train_fold(args, fold, tr, va, contexts, responses, y_mi, y_pg, device)


if __name__ == "__main__":
    main()


