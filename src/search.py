from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Protocol

import time
import torch


@dataclass(frozen=True)
class EpisodeContext:
    '''
    Stores permanent information about current episode:
    goal, steps_given, in-context examples (for the few-shot strategy only)
    '''
    goal: str
    steps_given: List[str]
    in_context: Optional[str] = None

@dataclass
class Candidate:
    task_name: str
    steps_pred: List[str] = field(default_factory=list)
    score: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

class PromptBuilder:
    def build(self, tasks: Dict[str, Any], cand: Candidate) -> str:
        
        parts = []
        
        task_cfg = tasks.get(cand.task_name, {})
        in_ctx_examples = task_cfg.get("in_context")
        if in_ctx_examples:
            parts.append(str(in_ctx_examples).strip() + "\n")

        parts.append(f"Task: {cand.task_name.strip()}")

        steps_txt = [f"Step {i}: {s.strip()}" for i, s in enumerate(cand.steps_pred, 1)]
        steps_txt.append(f"\nGive ONLY next Step {len(cand.steps_pred)+1}:")
        parts.append("; ".join(steps_txt))

        return "\n".join(parts)


class Proposer(Protocol):
    def propose(self, beam: List[Candidate], n: int) -> List[Candidate]:
        """Generate for each candidate in beam n branches."""
        ...


class Assessor(Protocol):
    def assess(self, candidates: List[Candidate]) -> List[Candidate]:
        """Calculate score/metrics for each candidate (in-place)."""
        ...


class Pruner(Protocol):
    def prune(self, candidates: List[Candidate], k: int) -> List[Candidate]:
        """Leave k best."""
        ...


# -----------------------------
# Default implementations
# -----------------------------

class Proposer:
    def __init__(
        self,
        llm_generate: Callable[[List[str], int], List[List[Dict[str, Any]]]],
        builder: PromptBuilder,
        mapper,  # obj with map_batch method
    ):
        self.llm_generate = llm_generate
        self.builder = builder
        self.mapper = mapper

    def propose(self, tasks: Dict[str, Any], beam: List[Candidate], n: int) -> List[Candidate]:
        # Build prompts for each candidate in beam (B prompts)
        prompts = [self.builder.build(tasks, cand) for cand in beam] 

        t0 = time.perf_counter()
        # Generate N samples per prompt -> List [B][n], each elem is dict {"new_gen_text", "V_G", ...}
        llm_out = self.llm_generate(prompts, n)
        t1 = time.perf_counter()
        print(f" llm_out: {(t1 - t0)*1000:.2f} ms")

        # Map + expand -> flat list of children length of lenth B*n
        t0 = time.perf_counter()
        children = self.mapper.map_and_expand(beam, llm_out, keep_empty=True)
        t1 = time.perf_counter()
        print(f"map and expand: {(t1 - t0)*1000:.2f} ms")

        return children


class Assessor:
    """
    Partial plan evaluation via YES/NO probability on the first generated token.

    - One prompt per candidate.
    - Uses tokenizer + model.generate(max_new_tokens=20, output_scores=True).
    - Stores V_P in candidate.meta["V_P"].
    """

    SYSTEM_TEXT = (
        'You are a user trying to achieve the “given task” step-by-step, and you have done '
        'some “finished steps”. Evaluate whether the “finished steps” are appropriate in a '
        'step-by-step manner and whether they make progress towards completing the task. '
        'No typos in the text below. Output “YES” or “NO”, followed by explanation. '
    )

    def __init__(
        self,
        model,
        tokenizer,
        device: Optional[str] = None,
        yes_str: str = "YES",
        no_str: str = "NO",
    ):
        self.model = model
        self.tokenizer = tokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        # LLaMA-style: pad = eos if missing
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- check YES/NO are single tokens ---
        yes_ids = self.tokenizer.encode(yes_str, add_special_tokens=False)
        no_ids = self.tokenizer.encode(no_str, add_special_tokens=False)

        if len(yes_ids) != 1 or len(no_ids) != 1:
            raise ValueError(
                f'Expected {yes_str!r} and {no_str!r} to be single tokens, but got '
                f'{yes_str!r} ids={yes_ids} (len={len(yes_ids)}), '
                f'{no_str!r} ids={no_ids} (len={len(no_ids)}).'
            )

        self.yes_token_id = yes_ids[0]
        self.no_token_id = no_ids[0]
        self.yes_str = yes_str
        self.no_str = no_str

    def _build_prompt(self, cand: Candidate) -> str:
        if cand.steps_pred:
            steps = "; ".join([f"Step {i+1}: {s}" for i, s in enumerate(cand.steps_pred)])
        else:
            steps = ""

        return (
            f"{self.SYSTEM_TEXT}\n\n"
            f"Given task: {cand.task_name}.\n"
            f"Finished steps: {steps}\n"
        )

    @torch.inference_mode()
    def assess(self, candidates: List[Candidate]) -> List[Candidate]:
        if not candidates:
            return candidates

        prompts = [self._build_prompt(c) for c in candidates]
        for i, p in enumerate(prompts):
            print("child {i+1} prompt: ", p)

        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        enc = enc.to(self.device)
        prompt_len_padded = enc["input_ids"].shape[1]

        out = self.model.generate(
            **enc,
            max_new_tokens=20,
            do_sample=False, 
            num_return_sequences=1,
            return_dict_in_generate=True,
            output_scores=True, 
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        # Scores for the first token generated
        logits0 = out.scores[0]

        l_yes = logits0[:, self.yes_token_id]
        l_no = logits0[:, self.no_token_id]
        print("YES: ", l_yes)
        print("NO: ", l_no)
        v_p = torch.softmax(torch.stack([l_yes, l_no], dim=1), dim=1)[:, 0]  # [B] softmax values for YES

        for i, c in enumerate(candidates):
            c.meta["V_P"] = float(v_p[i].item())

        gen_ids = out.sequences[:, prompt_len_padded:]  # [B, T_gen]
        gen_texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        for i, c in enumerate(candidates):
                print(c.meta["V_P"])
                print(gen_texts[i])
                print("=" * 80)

        return candidates


class TopKPruner:
    def prune(self, candidates: List[Candidate], k: int) -> List[Candidate]:
        return heapq.nlargest(k, candidates, key=lambda c: c.score)


class PromptAppender:
    """
    Вынесено отдельно, чтобы менять формат промпта
    (например, Goal/Steps so far/Next step).
    """
    def build_child(self, parent: Candidate, next_step: str) -> Candidate:
        next_step = next_step.strip()
        new_steps = parent.steps + [next_step]
        new_prompt = parent.prompt + "\n" + next_step
        return Candidate(prompt=new_prompt, steps=new_steps)


class LLMProposer:
    """
    Propose = для каждого кандидата сэмплим N вариантов у LLM.
    llm_generate(prompt, n) -> List[str]
    """
    def __init__(self, llm_generate, composer: Optional[PromptAppender] = None):
        self.llm_generate = llm_generate
        self.composer = composer or PromptAppender()

    def propose(self, beam: List[Candidate], n: int) -> List[Candidate]:
        out: List[Candidate] = []
        for cand in beam:
            gens = self.llm_generate(cand.prompt, n)
            for g in gens:
                out.append(self.composer.build_child(cand, g))
        return out


# -----------------------------
# Search engine
# -----------------------------
class BeamBFSSearch:
    """
    Твой цикл: Propose -> Assess -> Prune на каждом уровне.
    """
    def __init__(self, proposer: Proposer, assessor: Assessor, pruner: Pruner):
        self.proposer = proposer
        self.assessor = assessor
        self.pruner = pruner

    def run(self, start_prompt: str, depth: int, k: int, n: int) -> List[Candidate]:
        beam: List[Candidate] = [Candidate(prompt=start_prompt, steps=[])]
        for _lvl in range(depth):
            # 1) Propose
            candidates = self.proposer.propose(beam, n=n)
            if not candidates:
                break

            # 2) Assess (in-place)
            self.assessor.assess(candidates)

            # 3) Prune
            beam = self.pruner.prune(candidates, k=k)

        return beam