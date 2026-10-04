from typing import List, Optional
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class KyoCrossDecisionModel(nn.Module):
    """Pairwise Cross-Encoder с Dual Pooling (CLS + Masked Mean)."""

    def __init__(self, config: AutoConfig, torch_dtype: Optional[torch.dtype] = None):
        super().__init__()
        # Инициализация каркаса без загрузки сторонних весов
        self.encoder = AutoModel.from_config(config, attn_implementation="sdpa")
        hidden_size = config.hidden_size  # 384

        # Dual Pooling: CLS (384) + Mean (384) = 768
        self.ln = nn.LayerNorm(hidden_size * 2, dtype=torch_dtype or torch.bfloat16)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
        ).to(dtype=torch_dtype or torch.bfloat16)

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
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(last_hidden).float()
        sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        mean_rep = (sum_embeddings / sum_mask).to(dtype=cls_rep.dtype)

        # 3. LayerNorm + классификатор
        features = self.ln(torch.cat([cls_rep, mean_rep], dim=-1))
        flat_scores = self.classifier(features).squeeze(-1)

        # 4. Батчинг логитов по опциям
        batch_size = len(num_options)
        max_k = max(num_options)
        padded_logits = torch.full(
            (batch_size, max_k),
            -1e4,
            device=input_ids.device,
            dtype=cls_rep.dtype,
        )

        start_idx = 0
        for i, k in enumerate(num_options):
            end_idx = start_idx + k
            padded_logits[i, :k] = flat_scores[start_idx:end_idx]
            start_idx = end_idx

        return padded_logits
