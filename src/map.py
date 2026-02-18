from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    SentenceTransformer = None


@dataclass
class Candidate:
    # добавь сюда свои поля (ctx_key/task_id тоже можно)
    steps_pred: List[str] = field(default_factory=list)
    score: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


class ActionMapper:
    """
    Maps free-form LLM text to the most semantically similar admissible action using SBERT embeddings.
    V_M = max cosine similarity between Emb(text) and Emb(action).

    Usage:
        mapper = SBERTActionMapper(admissible_actions, model_name="all-MiniLM-L6-v2")
        children = mapper.map_and_expand(beam, llm_out)  # -> [B][n] Candidates
    """

    def __init__(
        self,
        admissible_actions: Sequence[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 64,
        normalize: bool = True,
    ):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is not installed. Install it with `pip install sentence-transformers` "
                "or tell me and I will provide a pure-transformers fallback."
            )

        self.admissible_actions: List[str] = list(admissible_actions)
        self.batch_size = batch_size
        self.normalize = normalize

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

        # Precompute action embeddings once
        self._action_emb = self._encode_texts(self.admissible_actions)  # tensor [A, D]
        if self.normalize:
            self._action_emb = self._l2_normalize(self._action_emb)     # cosine via dot

    @staticmethod
    def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return x / (x.norm(dim=1, keepdim=True) + eps)

    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        # SentenceTransformer.encode can directly return torch tensor
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        # emb: torch.Tensor [N, D] on self.device
        return emb

    @torch.no_grad()
    def map_batch(self, texts: List[str]) -> Tuple[List[str], List[float]]:
        """
        Returns:
            best_actions: list[str] length N
            v_m: list[float] length N (max cosine sim)
        """
        if not texts:
            return [], []

        txt_emb = self._encode_texts(texts)                 # [N, D]
        if self.normalize:
            txt_emb = self._l2_normalize(txt_emb)           # [N, D]

        # Cosine sim (if normalized) = dot product
        # sims: [N, A]
        sims = txt_emb @ self._action_emb.T

        best_idx = torch.argmax(sims, dim=1)                    # [N]
        best_sim = sims[torch.arange(sims.shape[0]), best_idx]  # [N]

        best_actions = [self.admissible_actions[i] for i in best_idx.tolist()]
        v_m = best_sim.detach().float().cpu().tolist()
        return best_actions, v_m

    @torch.no_grad()
    def map_and_expand(
        self,
        beam: List[Candidate],                 # B parents
        llm_out: List[List[Dict[str, Any]]],   # [B][n], each dict has {"new_gen_text":..., "V_G":...}
        keep_empty: bool = True,               # flag to keep alignment for block slicing
    ) -> List[Candidate]:
        """
        Flattened expansion:
        - assumes llm_out is exactly [B][n] in the same order as beam
        - flattens in row-major order: first n belong to beam[0], next n to beam[1], ...
        - maps each new_gen_text -> best admissible action and V_M (max cosine sim)
        - returns a flat list of children of length B*n

        If keep_empty=False, empty texts are skipped and alignment breaks; so this mode is not recommended
        if you rely on block structure.
        """
        B = len(beam)
        if B == 0:
            return []

        if len(llm_out) != B:
            raise ValueError(f"llm_out must have length B={B}, got {len(llm_out)}")

        # infer n from first parent; also validate rectangular shape
        n = len(llm_out[0])
        if n == 0:
            return []

        for i, samples in enumerate(llm_out):
            if len(samples) != n:
                raise ValueError(
                    f"llm_out must be rectangular [B][n]. Row {i} has len {len(samples)} != {n}."
                )

        # Flatten texts and V_G (makes sure we do NOT drop items if you need alignment)
        flat_texts: List[str] = []
        flat_vg: List[float] = []

        for samples in llm_out:
            for s in samples:
                text = str(s.get("new_gen_text", "")).strip()
                vg = float(s.get("V_G", 0.0))
                if not text and keep_empty:
                    text = ""  # keep placeholder to preserve order
                elif not text and not keep_empty:
                    # skipping breaks "first n belongs to first parent" alignment
                    continue

                flat_texts.append(text)
                flat_vg.append(vg)

        if not flat_texts:
            return []

        # map texts -> best action + V_M (same length as flat_texts)
        best_actions, v_m = self.map_batch(flat_texts)
        print("MAPPED:", best_actions, "TEXTS", flat_texts)
        # build flat children list
        children: List[Candidate] = []

        for idx, (text, vg, act, vm) in enumerate(zip(flat_texts, flat_vg, best_actions, v_m)):
            parent_i = idx // n  # block mapping: first candidate parent -> 0, next -> 1, ...
            parent = beam[parent_i]

            child = Candidate(
                steps_pred=list(parent.steps_pred) + [act],
                score=parent.score,
                meta=dict(parent.meta),
            )
            child.meta.update({
                "last_raw": text,
                "mapped_action": act,
                "V_G": vg, # We need to store value of V_G which came from llm_generate 
                "V_M": float(vm),
            })
            children.append(child)

        return children

