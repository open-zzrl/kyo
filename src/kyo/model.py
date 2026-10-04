from typing import List, Optional
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class KyoCrossDecisionModel(nn.Module):
    """Pairwise Cross-Encoder с Dual Pooling (CLS + Masked Mean)."""

    def __init__(self, config: AutoConfig, torch_dtype: Optional[torch.dtype] = None):
        super().__init__()
        dtype = torch_dtype or (
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )

        self.encoder = AutoModel.from_config(config, attn_implementation="sdpa")
        hidden_size = config.hidden_size  # 384

        self.ln = nn.LayerNorm(hidden_size * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
        )

        # Кастуем все модули к единому типу
        self.to(dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_options: List[int],
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state

        # 1. Токен [CLS]
        cls_rep = last_hidden[:, 0, :]

        # 2. Masked Mean Pooling
        mask_expanded = (
            attention_mask.unsqueeze(-1).expand_as(last_hidden).to(last_hidden.dtype)
        )
        sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        mean_rep = sum_embeddings / sum_mask

        # 3. Гарантированный каст к типу весов LayerNorm
        raw_features = torch.cat([cls_rep, mean_rep], dim=-1).to(
            self.ln.weight.dtype
        )
        features = self.ln(raw_features)
        flat_scores = self.classifier(features).squeeze(-1)

        # 4. Батчинг логитов
        batch_size = len(num_options)
        max_k = max(num_options)
        padded_logits = torch.full(
            (batch_size, max_k),
            -1e4,
            device=input_ids.device,
            dtype=flat_scores.dtype,
        )

        start_idx = 0
        for i, k in enumerate(num_options):
            end_idx = start_idx + k
            padded_logits[i, :k] = flat_scores[start_idx:end_idx]
            start_idx = end_idx

        return padded_logits
