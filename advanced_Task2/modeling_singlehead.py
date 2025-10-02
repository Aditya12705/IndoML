from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


@dataclass
class SingleHeadConfig:
    backbone_name: str = "models/bert-base-uncased"
    num_labels: int = 3
    dropout: float = 0.1


class SingleHeadClassifier(nn.Module):
    def __init__(self, cfg: SingleHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        hf_cfg = AutoConfig.from_pretrained(cfg.backbone_name)
        self.encoder = AutoModel.from_pretrained(cfg.backbone_name, config=hf_cfg)
        hidden = getattr(hf_cfg, "hidden_size", 768)
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(hidden, cfg.num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            last = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            summed = (last * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-6)
            pooled = summed / denom

        x = self.dropout(pooled)
        logits = self.head(x)
        res = {"logits": logits}
        if labels is not None:
            if class_weights is not None:
                loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            else:
                loss_fn = nn.CrossEntropyLoss()
            res["loss"] = loss_fn(logits, labels)
        return res


