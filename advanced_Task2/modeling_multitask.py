from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


@dataclass
class MultiTaskConfig:
    backbone_name: str = "models/bert-base-uncased"
    num_labels_mi: int = 3
    num_labels_pg: int = 3
    dropout: float = 0.1


class MultiTaskClassifier(nn.Module):
    def __init__(
        self,
        config: MultiTaskConfig,
    ) -> None:
        super().__init__()
        self.config = config
        hf_config = AutoConfig.from_pretrained(config.backbone_name)
        self.encoder = AutoModel.from_pretrained(config.backbone_name, config=hf_config)
        hidden_size = getattr(hf_config, "hidden_size", 768)
        self.dropout = nn.Dropout(config.dropout)
        self.head_mi = nn.Linear(hidden_size, config.num_labels_mi)
        self.head_pg = nn.Linear(hidden_size, config.num_labels_pg)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels_mi: Optional[torch.Tensor] = None,
        labels_pg: Optional[torch.Tensor] = None,
        loss_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
        )
        # Use pooled representation: last_hidden_state CLS or mean-pool if no pooler
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            # Mean pool over attention mask
            last_hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            summed = (last_hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-6)
            pooled = summed / denom

        x = self.dropout(pooled)
        logits_mi = self.head_mi(x)
        logits_pg = self.head_pg(x)

        result: Dict[str, torch.Tensor] = {
            "logits_mi": logits_mi,
            "logits_pg": logits_pg,
        }

        if labels_mi is not None and labels_pg is not None:
            # Class-weighted CE if provided
            if loss_weights is not None:
                w_mi, w_pg = loss_weights
                loss_fn_mi = nn.CrossEntropyLoss(weight=w_mi)
                loss_fn_pg = nn.CrossEntropyLoss(weight=w_pg)
            else:
                loss_fn_mi = nn.CrossEntropyLoss()
                loss_fn_pg = nn.CrossEntropyLoss()
            loss = loss_fn_mi(logits_mi, labels_mi) + loss_fn_pg(logits_pg, labels_pg)
            result["loss"] = loss

        return result


