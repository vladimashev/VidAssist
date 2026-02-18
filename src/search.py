from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Protocol

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
        steps_txt.append(f"\nStep {len(cand.steps_pred)+1}:")
        parts.append("; ".join(steps_txt))

        return "\n".join(parts)


class Proposer(Protocol):
    def propose(self, beam: List[Candidate], n: int) -> List[Candidate]:
        """Generate for each candidate in beam n branches."""
        ...


class Assessor(Protocol):
    def assess(self, candidates: List[Candidate]) -> None:
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

        # Generate N samples per prompt -> List [B][n], each elem is dict {"new_gen_text", "V_G", ...}
        llm_out = self.llm_generate(prompts, n)

        # Map + expand -> flat list of children length of lenth B*n
        children = self.mapper.map_and_expand(beam, llm_out, keep_empty=True)

        return children


class TopKPruner:
    def prune(self, candidates: List[Candidate], k: int) -> List[Candidate]:
        return heapq.nlargest(k, candidates, key=lambda c: c.score)


class SimpleAssessor:
    """
     
    """
    def assess(self, candidates: List[Candidate]) -> None:
        for c in candidates:
            text = c.steps[-1] if c.steps else ""
            m1 = float("boil" in text.lower() or "heat" in text.lower())
            m2 = -0.01 * len(text)
            c.metrics = {"keyword_bonus": m1, "len_penalty": m2}
            c.score = m1 + m2


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