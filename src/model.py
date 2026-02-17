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

        B = enc["input_ids"].shape[0] # B = len(prompts), i.e. number of input candidates
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

        sequences = out.sequences  # tensor [B*n, T_total], where T_total = T_old (inc padding) + T_new
        print(sequences)
        scores = out.scores        # list of length T_new; each is of shape [B*n, vocab]
        T_new = len(scores)
        Bn = B * n

        # Prompt lengths
        T_old = enc["input_ids"].shape[1]                                       # Prompt length with padding
        prompt_lens_no_pad = enc["attention_mask"].sum(dim=1)                   # [B]
        prompt_lens__no_pad_rep = prompt_lens_no_pad.repeat_interleave(n)       # [B*n]

        # Calculate logprob for each token generated
        # New token logprobs [B*n, T_new]
        new_token_logprobs = torch.empty((Bn, T_new), device=sequences.device, dtype=torch.float32)
        row = torch.arange(Bn, device=sequences.device)

        for t in range(T_new):
            logits_t = scores[t]  # [B*n, vocab]
            log_probs_t = torch.log_softmax(logits_t, dim=-1) # Calculate distribution

            token_ids_t = sequences[row, T_old + t]  # [B*n] - next tokens generated in each 
            new_token_logprobs[:, t] = log_probs_t.gather(1, token_ids_t.unsqueeze(1)).squeeze(1)

        results: List[List[Dict[str, Any]]] = [[] for _ in range(B)]

        gen_ids_all = sequences[:, T_old:T_old + T_new]   # [Bn, T_new]

        # Find cut positions t_end for each sample
        if eos_id is not None:
            eos_mask = (gen_ids_all == eos_id)            # [Bn, T_new]
            has_eos = eos_mask.any(dim=1)                 # [Bn]
            first_eos = eos_mask.float().argmax(dim=1)    # [Bn] (index of first True if any)

            t_end = torch.where(has_eos, first_eos, torch.full_like(first_eos, T_new))
            if not self.exclude_eos:
                t_end = torch.clamp(t_end + 1, max=T_new)
        else:
            t_end = torch.full((Bn,), T_new, device=sequences.device, dtype=torch.long)

        # Keep only tokens before t_end
        t_idx = torch.arange(T_new, device=sequences.device).unsqueeze(0)  # [1, T_new]
        keep = t_idx < t_end.unsqueeze(1)                                  # [Bn, T_new]

        n_tokens = keep.sum(dim=1)                                         # [Bn]
        sum_logprob = (new_token_logprobs * keep).sum(dim=1)               # [Bn]
        v_g = sum_logprob / n_tokens.clamp(min=1)                          # [Bn]

        # set empty generations to -inf
        empty = (n_tokens == 0)
        sum_logprob = sum_logprob.masked_fill(empty, float("-inf"))
        v_g = v_g.masked_fill(empty, float("-inf"))

        # --- pack back to [B][n] and decode text ---
        results = [[] for _ in range(B)]
        for idx in range(Bn):
            cut = int(t_end[idx].item())
            ids = gen_ids_all[idx, :cut]
            text = "" if cut == 0 else self.tokenizer.decode(ids, skip_special_tokens=True).strip()

            results[idx // n].append({
                "text": text,
                "V_G": float(v_g[idx].item()),
                "sum_logprob": float(sum_logprob[idx].item()),
                "n_tokens": int(n_tokens[idx].item()),
            })
        return results