from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch


@dataclass
class GenConfig:
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True


class HFLLMGenerate:
    """
    llm_generate(prompts, n) -> [B][N] dicts:
      {"text": ..., "V_G": ..., "sum_logprob": ..., "n_tokens": ...}

    V_G = mean log prob of generated tokens (excluding prompt tokens and excluding EOS by default).
    """
    def __init__(self, model, tokenizer, cfg: Optional[GenConfig] = None, exclude_eos: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg or GenConfig()
        self.exclude_eos = exclude_eos

        # For LLaMA eos as pad
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def __call__(self, prompts: List[str], n: int) -> List[List[Dict[str, Any]]]:
        if not prompts:
            return []

        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        B = enc["input_ids"].shape[0]
        eos_id = self.tokenizer.eos_token_id

        out = self.model.generate(
            **enc,
            do_sample=self.cfg.do_sample,
            num_return_sequences=n,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=eos_id,
            return_dict_in_generate=True,
            output_scores=True,  # for logprobs calculation
        )

        sequences = out.sequences  # [B*n, T_total]
        scores = out.scores        # list of length T_new; each is of shape [B*n, vocab]
        T_new = len(scores)
        Bn = B * n

        # длины промптов (без паддинга)
        prompt_lens = enc["attention_mask"].sum(dim=1)              # [B]
        prompt_lens_rep = prompt_lens.repeat_interleave(n)          # [B*n]

        # Считаем logprob выбранного токена на каждом шаге генерации
        # new_token_logprobs: [B*n, T_new]
        new_token_logprobs = torch.empty((Bn, T_new), device=sequences.device, dtype=torch.float32)

        for t in range(T_new):
            logits_t = scores[t]  # [B*n, vocab]
            log_probs_t = torch.log_softmax(logits_t, dim=-1)

            
            row = torch.arange(Bn, device=sequences.device)
            for t in range(T_new):
                logits_t = scores[t]  # [B*n, vocab]
                log_probs_t = torch.log_softmax(logits_t, dim=-1)

                token_ids_t = sequences[row, prompt_lens_rep + t]  # [B*n]
                new_token_logprobs[:, t] = log_probs_t.gather(1, token_ids_t.unsqueeze(1)).squeeze(1)

        results: List[List[Dict[str, Any]]] = [[] for _ in range(B)]

        for i in range(B):
            for j in range(n):
                idx = i * n + j
                start = int(prompt_lens_rep[idx].item())

                # Сгенерированные токены (максимум T_new)
                gen_ids = sequences[idx, start:start + T_new]  # [T_new]
                lp = new_token_logprobs[idx]                   # [T_new]

                # Обрезаем по EOS (обычно EOS не включают в nk)
                t_end = T_new
                if eos_id is not None:
                    eos_pos = (gen_ids == eos_id).nonzero(as_tuple=False)
                    if eos_pos.numel() > 0:
                        first_eos = int(eos_pos[0].item())
                        t_end = first_eos if self.exclude_eos else first_eos + 1

                gen_ids_trim = gen_ids[:t_end]
                lp_trim = lp[:t_end]

                n_tokens = int(lp_trim.numel())
                if n_tokens == 0:
                    # пустая генерация — можно либо пропустить, либо вернуть очень плохой скор
                    text = ""
                    sum_logprob = float("-inf")
                    v_g = float("-inf")
                else:
                    text = self.tokenizer.decode(gen_ids_trim, skip_special_tokens=True).strip()
                    sum_logprob = float(lp_trim.sum().item())
                    v_g = sum_logprob / n_tokens

                results[i].append({
                    "text": text,
                    "V_G": v_g,
                    "sum_logprob": sum_logprob,
                    "n_tokens": n_tokens,
                })

        return results