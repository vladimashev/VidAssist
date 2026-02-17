cofrom dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Protocol
import heapq


@dataclass
class Candidate:
    prompt: str
    steps: List[str] = field(default_factory=list)
    score: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)  # details assess
    meta: Dict[str, Any] = field(default_factory=dict)       # the rest (ids, logits, etc.)


@dataclass(frozen=True)
class EpisodeContext:
    in_context: Optional[str]      # None for zero-shot
    goal: str
    observation: Optional[str] = None

@dataclass
class Candidate:
    steps_pred: List[str] = field(default_factory=list)
    score: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

class PromptBuilder:
    def __init__(self, step_prefix: str = "Step"):
        self.step_prefix = step_prefix

    def build(self, ctx: EpisodeContext, steps_so_far: List[str]) -> str:
        parts = []
        if ctx.in_context:
            parts.append(ctx.in_context.strip())
        # “Goal description”
        parts.append(f"Task: {ctx.goal.strip()}")
        if ctx.observation:
            parts.append(ctx.observation.strip())

        # “Previous plan”
        # В твоём примере это: Task: ... Step 1: ... Step 2: ... Step k:
        steps_txt = []
        for i, s in enumerate(steps_so_far, 1):
            steps_txt.append(f"{self.step_prefix} {i}: {s.strip()}")
        steps_txt.append(f"{self.step_prefix} {len(steps_so_far)+1}:")  # запрос на следующий шаг

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