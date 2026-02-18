from src.dataset import CrossTaskVideoDataset
from src.search import Candidate, Proposer, PromptBuilder
from src.model import HFLLMGenerate, GenConfig
from src.map import ActionMapper
from transformers import AutoTokenizer, AutoModelForCausalLM

dataset = CrossTaskVideoDataset(data_root="data/crosstask")
sample = dataset[0]

cand = Candidate(
    task_name=sample["task_name"],
    steps_pred=sample["pred_sequence"][:2],
    score=0.0,
    meta={},
)
beam = [cand]

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

llm_generate = HFLLMGenerate(model, tokenizer, cfg=GenConfig(max_new_tokens=10))

mapper = ActionMapper(admissible_actions=dataset.admissible_actions)
builder = PromptBuilder()
proposer = Proposer(llm_generate=llm_generate, builder=builder, mapper=mapper)

children = proposer.propose(tasks=dataset.tasks, beam=beam, n=5)

print("children:", len(children))
for i, c in enumerate(children[:5]):
    print(i, c.steps_pred, c.meta.get("V_G"), c.meta.get("V_M"), c.meta.get("last_raw"))
