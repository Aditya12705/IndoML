import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score # Import f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# It's good practice to place your custom model definitions in a separate file,
# but for simplicity, assuming it's in the same directory or accessible.
from .modeling_singlehead import SingleHeadConfig, SingleHeadClassifier


# --- NEW: Focal Loss Implementation ---
class FocalLoss(nn.Module):
    """
    Focal Loss class for handling class imbalance.
    Reduces the loss for well-classified examples, forcing the model to focus on harder examples.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        # alpha is the weighting factor for each class, similar to class_weights
        self.alpha = alpha
        # gamma is the focusing parameter
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs are the raw logits from the model
        # targets are the ground truth labels
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss) # Probability of the correct class
        focal_loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --- Utility Functions (Unchanged) ---
def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_pair(context: str, response: str) -> Tuple[str, str]:
    return context, response

def extract_examples(train_items: List[Dict], track: str) -> Tuple[List[str], List[str], List[int]]:
    label_map = {"No": 0, "To some extent": 1, "Yes": 2}
    contexts: List[str] = []
    responses: List[str] = []
    labels: List[int] = []
    key = "Mistake_Identification" if track == "mi" else "Providing_Guidance"
    for item in train_items:
        ctx = item.get("conversation_history", "")
        for _name, resp in item.get("tutor_responses", {}).items():
            ann = resp.get("annotation", {})
            if key in ann:
                contexts.append(ctx)
                responses.append(resp.get("response", ""))
                labels.append(label_map[ann[key]])
    return contexts, responses, labels

# --- Dataset Class (Unchanged) ---
class PairDataset(Dataset):
    def __init__(self, contexts: List[str], responses: List[str], labels: List[int], tokenizer, max_len: int) -> None:
        self.contexts = contexts
        self.responses = responses
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict:
        context, response = build_pair(self.contexts[idx], self.responses[idx])
        inputs = self.tokenizer.encode_plus(
            context,
            response,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# --- MODIFIED: train_loop function ---
def train_loop(cfg, dl_tr, dl_va, fold: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model_cfg = SingleHeadConfig(model_name=cfg.model_name, num_classes=3)
    model = SingleHeadClassifier(model_cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    num_training_steps = math.ceil(len(dl_tr) / cfg.grad_accum) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    labels_tr = [item["labels"].item() for item in dl_tr.dataset]
    n_samples = len(labels_tr)
    class_counts = [labels_tr.count(c) for c in range(3)]
    class_weights = [n_samples / (3 * count) if count > 0 else 1 for count in class_counts]
    
    # MODIFICATION: Use FocalLoss instead of CrossEntropyLoss
    loss_fn = FocalLoss(alpha=torch.tensor(class_weights, device=device, dtype=torch.float), gamma=2.0)

    # MODIFICATION: Track best F1 score instead of loss
    best_f1 = 0.0
    ckpt = os.path.join(cfg.out_dir, f"model_{cfg.track}_fold{fold}.pt")

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        for i, batch in enumerate(dl_tr):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            outputs = model(batch)
            loss = loss_fn(outputs, labels)
            loss = loss / cfg.grad_accum
            loss.backward()
            train_loss += loss.item()

            if (i + 1) % cfg.grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(dl_tr):.4f}")

        # --- MODIFICATION: Validation and Model Saving Logic ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dl_va:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                outputs = model(batch)
                
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= max(1, len(dl_va))
        
        # Calculate Macro F1 score
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        
        print(f"Epoch {epoch + 1}, Val Loss: {val_loss:.4f}, Val Macro F1: {macro_f1:.4f}")

        # Save the model based on the best F1 score
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            print(f"🎉 New best Macro F1: {best_f1:.4f}. Saving model to {ckpt}")
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, ckpt)


# --- Main Function (Unchanged) ---
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--track", choices=["mi", "pg"], required=True, help="mi=Mistake_Identification, pg=Providing_Guidance")
    p.add_argument("--model_name", default="microsoft/deberta-v3-large") # Using a strong baseline
    p.add_argument("--train_path", default=os.path.join("IndoML_Datathon", "data", "trainset.json"))
    p.add_argument("--out_dir", default=os.path.join("IndoML_Datathon", "advanced_Task2", "models"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5) # Smaller LR for larger models
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--grad_accum", type=int, default=1)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)
    train_items = read_json(args.train_path)
    contexts, responses, labels = extract_examples(train_items, args.track)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(contexts, labels)):
        print(f"\n--- Fold {fold} ---")
        tr_contexts, va_contexts = [contexts[i] for i in train_idx], [contexts[i] for i in val_idx]
        tr_responses, va_responses = [responses[i] for i in train_idx], [responses[i] for i in val_idx]
        tr_labels, va_labels = [labels[i] for i in train_idx], [labels[i] for i in val_idx]

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        ds_tr = PairDataset(tr_contexts, tr_responses, tr_labels, tokenizer, args.max_len)
        ds_va = PairDataset(va_contexts, va_responses, va_labels, tokenizer, args.max_len)
        dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True)
        dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False)

        train_loop(args, dl_tr, dl_va, fold)

if __name__ == "__main__":
    main()