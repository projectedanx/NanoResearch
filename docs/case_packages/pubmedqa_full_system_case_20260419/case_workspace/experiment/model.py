import logging
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@dataclass
class PubMedBERTOutput:
    logits: torch.Tensor
    cls_embedding: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


class ContrastiveLossModule(nn.Module):
    """
    Cosine-similarity based contrastive loss.
    - If negatives are not provided: uses in-batch negatives (InfoNCE, symmetric).
    - If negatives are provided: computes positive-vs-negative logits per sample.
    """

    def __init__(self, temperature: float = 0.07, eps: float = 1e-8) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = temperature
        self.eps = eps

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if anchor_embeddings.ndim != 2 or positive_embeddings.ndim != 2:
            raise ValueError("anchor_embeddings and positive_embeddings must be 2D tensors [B, D].")
        if anchor_embeddings.shape != positive_embeddings.shape:
            raise ValueError("anchor_embeddings and positive_embeddings must have same shape [B, D].")

        anchor = F.normalize(anchor_embeddings, p=2, dim=-1, eps=self.eps)
        positive = F.normalize(positive_embeddings, p=2, dim=-1, eps=self.eps)

        # In-batch negatives
        if negative_embeddings is None:
            logits = torch.matmul(anchor, positive.transpose(0, 1)) / self.temperature  # [B, B]
            targets = torch.arange(anchor.size(0), device=anchor.device)
            loss_i = F.cross_entropy(logits, targets)
            loss_j = F.cross_entropy(logits.transpose(0, 1), targets)
            return 0.5 * (loss_i + loss_j)

        # Explicit negatives
        if negative_embeddings.ndim not in (2, 3):
            raise ValueError("negative_embeddings must be [N, D] or [B, K, D].")

        if negative_embeddings.ndim == 2:
            negative = F.normalize(negative_embeddings, p=2, dim=-1, eps=self.eps)  # [N, D]
            pos_logits = torch.sum(anchor * positive, dim=-1, keepdim=True) / self.temperature  # [B, 1]
            neg_logits = torch.matmul(anchor, negative.transpose(0, 1)) / self.temperature  # [B, N]
            logits = torch.cat([pos_logits, neg_logits], dim=1)  # [B, 1+N]
            targets = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
            return F.cross_entropy(logits, targets)

        # [B, K, D]
        if negative_embeddings.size(0) != anchor.size(0):
            raise ValueError(
                "For negative_embeddings with shape [B, K, D], first dimension must match batch size."
            )
        negative = F.normalize(negative_embeddings, p=2, dim=-1, eps=self.eps)
        pos_logits = torch.sum(anchor * positive, dim=-1, keepdim=True) / self.temperature  # [B, 1]
        neg_logits = torch.einsum("bd,bkd->bk", anchor, negative) / self.temperature  # [B, K]
        logits = torch.cat([pos_logits, neg_logits], dim=1)  # [B, 1+K]
        targets = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
        return F.cross_entropy(logits, targets)


class PubMedBERTClassifier(nn.Module):
    """
    PubMedBERT encoder + classification head.
    Returns logits and CLS embedding for CE/consistency/contrastive objectives.
    """

    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int = 3,
        dropout: Optional[float] = None,
        pooling: str = "cls",
        output_hidden_states: bool = False,
    ) -> None:
        super().__init__()
        if pooling not in ("cls", "mean"):
            raise ValueError("pooling must be one of {'cls', 'mean'}")

        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.pooling = pooling
        self.output_hidden_states = output_hidden_states

        try:
            config = AutoConfig.from_pretrained(
                model_name_or_path,
                num_labels=num_labels,
                output_hidden_states=output_hidden_states,
            )
            self.encoder = AutoModel.from_pretrained(
                model_name_or_path,
                config=config,
                ignore_mismatched_sizes=True,  # required for robustness across checkpoints
            )
        except Exception as exc:
            LOGGER.exception("Failed to load model/config from '%s'", model_name_or_path)
            raise RuntimeError(f"Could not initialize model from {model_name_or_path}") from exc

        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            raise RuntimeError("Loaded config does not provide hidden_size.")

        classifier_dropout = (
            dropout
            if dropout is not None
            else getattr(config, "classifier_dropout", None)
            if getattr(config, "classifier_dropout", None) is not None
            else getattr(config, "hidden_dropout_prob", 0.1)
        )

        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self._init_classifier()

        LOGGER.info(
            "Initialized PubMedBERTClassifier | model=%s | num_labels=%d | pooling=%s | dropout=%.4f",
            model_name_or_path,
            num_labels,
            pooling,
            classifier_dropout,
        )

    def _init_classifier(self) -> None:
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def freeze_encoder(self, freeze: bool = True) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = not freeze
        LOGGER.info("Encoder frozen=%s", freeze)

    def _pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden_state[:, 0]

        # mean pooling
        if attention_mask is None:
            return last_hidden_state.mean(dim=1)

        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = torch.sum(last_hidden_state * mask, dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1e-8)
        return summed / denom

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            output_hidden_states=self.output_hidden_states,
        )
        return self._pool(outputs.last_hidden_state, attention_mask=attention_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[PubMedBERTOutput, Tuple[torch.Tensor, torch.Tensor]]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            output_hidden_states=self.output_hidden_states,
        )
        cls_embedding = self._pool(outputs.last_hidden_state, attention_mask=attention_mask)
        logits = self.classifier(self.dropout(cls_embedding))

        if return_dict:
            return PubMedBERTOutput(
                logits=logits,
                cls_embedding=cls_embedding,
                hidden_states=outputs.hidden_states if self.output_hidden_states else None,
                attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
            )
        return logits, cls_embedding


def compute_cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must be [B, C].")
    if labels.ndim != 1:
        raise ValueError("labels must be [B].")
    return F.cross_entropy(logits, labels)


def compute_consistency_kl_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    temperature: float = 1.0,
    symmetric: bool = True,
) -> torch.Tensor:
    """
    KL consistency between two predictive distributions.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if logits_a.shape != logits_b.shape:
        raise ValueError("logits_a and logits_b must have identical shape [B, C].")

    log_p = F.log_softmax(logits_a / temperature, dim=-1)
    q = F.softmax(logits_b / temperature, dim=-1)
    kl_ab = F.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)

    if not symmetric:
        return kl_ab

    log_q = F.log_softmax(logits_b / temperature, dim=-1)
    p = F.softmax(logits_a / temperature, dim=-1)
    kl_ba = F.kl_div(log_q, p, reduction="batchmean") * (temperature ** 2)
    return 0.5 * (kl_ab + kl_ba)


def compute_joint_cct_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    logits_pair: Optional[torch.Tensor] = None,
    emb: Optional[torch.Tensor] = None,
    emb_pair: Optional[torch.Tensor] = None,
    negative_emb: Optional[torch.Tensor] = None,
    lambda_consistency: float = 0.5,
    lambda_contrastive: float = 0.0,
    kl_temperature: float = 1.0,
    contrastive_temperature: float = 0.07,
) -> Dict[str, torch.Tensor]:
    """
    Composite loss for CCT-like training.
    Returns dict with total/ce/consistency/contrastive.
    """
    if lambda_consistency < 0 or lambda_contrastive < 0:
        raise ValueError("lambda_consistency and lambda_contrastive must be >= 0")

    ce_loss = compute_cross_entropy_loss(logits, labels)
    total_loss = ce_loss

    consistency_loss = torch.zeros((), device=logits.device)
    if logits_pair is not None and lambda_consistency > 0:
        consistency_loss = compute_consistency_kl_loss(
            logits, logits_pair, temperature=kl_temperature, symmetric=True
        )
        total_loss = total_loss + lambda_consistency * consistency_loss

    contrastive_loss = torch.zeros((), device=logits.device)
    if emb is not None and emb_pair is not None and lambda_contrastive > 0:
        contrastive_fn = ContrastiveLossModule(temperature=contrastive_temperature)
        contrastive_loss = contrastive_fn(emb, emb_pair, negative_embeddings=negative_emb)
        total_loss = total_loss + lambda_contrastive * contrastive_loss

    return {
        "total_loss": total_loss,
        "ce_loss": ce_loss.detach(),
        "consistency_loss": consistency_loss.detach(),
        "contrastive_loss": contrastive_loss.detach(),
    }


def create_model(
    model_name_or_path: str,
    num_labels: int = 3,
    dropout: Optional[float] = None,
    pooling: str = "cls",
    output_hidden_states: bool = False,
    device: Optional[Union[str, torch.device]] = None,
) -> PubMedBERTClassifier:
    model = PubMedBERTClassifier(
        model_name_or_path=model_name_or_path,
        num_labels=num_labels,
        dropout=dropout,
        pooling=pooling,
        output_hidden_states=output_hidden_states,
    )
    if device is not None:
        model = model.to(device)
    return model


__all__ = [
    "PubMedBERTOutput",
    "ContrastiveLossModule",
    "PubMedBERTClassifier",
    "compute_cross_entropy_loss",
    "compute_consistency_kl_loss",
    "compute_joint_cct_loss",
    "create_model",
]


if __name__ == "__main__":
    # Lightweight smoke test for local sanity checks.
    try:
        import argparse

        parser = argparse.ArgumentParser(description="Smoke test PubMedBERTClassifier")
        parser.add_argument(
            "--model_name_or_path",
            type=str,
            required=True,
            help="Path or HF id for PubMedBERT/BioBERT model.",
        )
        parser.add_argument("--seq_len", type=int, default=16)
        parser.add_argument("--batch_size", type=int, default=2)
        parser.add_argument("--num_labels", type=int, default=3)
        args = parser.parse_args()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_model(
            model_name_or_path=args.model_name_or_path,
            num_labels=args.num_labels,
            device=device,
        )
        model.eval()

        input_ids = torch.ones((args.batch_size, args.seq_len), dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        print(
            {
                "logits_shape": tuple(out.logits.shape),
                "cls_embedding_shape": tuple(out.cls_embedding.shape),
            }
        )
    except Exception:
        LOGGER.exception("Smoke test failed.")
        sys.exit(1)