import json
import os
import time
from typing import Any, Dict, List, Optional, Union

from huggingface_hub import snapshot_download
from safetensors.torch import load_file
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

from .model import KyoCrossDecisionModel
from .schemas import DecisionResult


class Kyo:
    """Kyo System-One Decision Engine."""

    def __init__(
        self,
        model: KyoCrossDecisionModel,
        tokenizer: AutoTokenizer,
        device: Optional[str] = None,
        confidence_threshold: float = 0.85,
        max_seq_len: int = 4096,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.confidence_threshold = confidence_threshold
        self.max_seq_len = max_seq_len

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "open-zzrl/kyo",
        device: Optional[str] = None,
        confidence_threshold: float = 0.85,
        max_seq_len: int = 4096,
        base_encoder_id: str = "jhu-clsp/mmBERT-small",
        **kwargs,
    ) -> "Kyo":
        """Загрузка движка с автоматическим разрешением конфига и весов."""
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        target_dtype = torch.bfloat16 if "cuda" in target_device else torch.float32

        if os.path.isdir(model_name_or_path):
            model_dir = model_name_or_path
        else:
            model_dir = snapshot_download(repo_id=model_name_or_path)

        # 1. Загрузка токенизатора с безопасным фоллбэком
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(base_encoder_id)

        # 2. Загрузка конфига с обработкой отсутствующего model_type
        config = None
        try:
            config = AutoConfig.from_pretrained(model_dir)
        except Exception:
            cfg_path = os.path.join(model_dir, "config.json")
            if os.path.exists(cfg_path):
                try:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg_dict = json.load(f)
                    if "model_type" not in cfg_dict:
                        cfg_dict["model_type"] = "modernbert"
                    config = AutoConfig.for_model(**cfg_dict)
                except Exception:
                    pass

        if config is None:
            config = AutoConfig.from_pretrained(base_encoder_id)

        # 3. Инициализация модели и загрузка весов
        model = KyoCrossDecisionModel(config=config, torch_dtype=target_dtype)

        sf_path = os.path.join(model_dir, "model.safetensors")
        bin_path = os.path.join(model_dir, "pytorch_model.bin")

        if os.path.exists(sf_path):
            state_dict = load_file(sf_path)
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(
                f"Файлы весов (model.safetensors / pytorch_model.bin) не найдены в {model_dir}"
            )

        model.load_state_dict({k: v.to(target_dtype) for k, v in state_dict.items()})

        return cls(
            model=model,
            tokenizer=tokenizer,
            device=target_device,
            confidence_threshold=confidence_threshold,
            max_seq_len=max_seq_len,
            **kwargs,
        )

    @torch.inference_mode()
    def decide(
        self,
        context: Union[str, Dict[str, Any]],
        instruction: str,
        options: Union[Dict[str, str], List[tuple]],
        threshold: Optional[float] = None,
        max_seq_len: Optional[int] = None,
    ) -> DecisionResult:
        t0 = time.perf_counter()

        if isinstance(context, dict):
            context_str = json.dumps(context, ensure_ascii=False, indent=2)
        else:
            context_str = str(context).strip()

        if isinstance(options, dict):
            opt_items = list(options.items())
        else:
            opt_items = list(options)

        if not opt_items:
            raise ValueError("Список options не может быть пустым.")

        keys = [item[0] for item in opt_items]
        num_options = [len(opt_items)]

        all_text_a = []
        all_text_b = []
        is_score = any(k in ["0", "1", "2", "3"] for k in keys)

        for k, desc in opt_items:
            prefix = (
                f"Candidate Option [Level {k}]"
                if is_score
                else f"Candidate Option ({k})"
            )
            all_text_a.append(f"{context_str}\n\nInstruction: {instruction.strip()}")
            all_text_b.append(f"{prefix}: {str(desc).strip()}")

        effective_max_len = max_seq_len or self.max_seq_len

        tok = self.tokenizer(
            all_text_a,
            all_text_b,
            padding=True,
            truncation=True,
            max_length=effective_max_len,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(tok["input_ids"], tok["attention_mask"], num_options)
        valid_logits = logits[0, : len(opt_items)].float()
        probs = F.softmax(valid_logits, dim=-1)

        conf, chosen_idx = torch.max(probs, dim=-1)
        conf_val = float(conf.item())
        chosen_key = keys[int(chosen_idx.item())]

        scores_dict = {keys[i]: float(probs[i].item()) for i in range(len(keys))}
        latency = (time.perf_counter() - t0) * 1000.0

        applied_threshold = (
            threshold if threshold is not None else self.confidence_threshold
        )
        fallback = conf_val < applied_threshold

        return DecisionResult(
            decision=chosen_key,
            confidence=conf_val,
            scores=scores_dict,
            fallback_to_llm=fallback,
            latency_ms=latency,
            raw_payload=context if isinstance(context, dict) else None,
        )
