from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import torch

# class StopOnTokens(StoppingCriteria):
#     def __init__(self, stop_token_ids, prompt_len):
#         self.stop_token_ids = set(stop_token_ids)
#         self.prompt_len = prompt_len

#     def __call__(self, input_ids, scores, **kwargs):
#         gen_ids = input_ids[0, self.prompt_len:]
#         return len(gen_ids) > 0 and gen_ids[-1].item() in self.stop_token_ids

# model_name = "meta-llama/Llama-2-7b-hf"
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# if tokenizer.pad_token_id is None:
#     tokenizer.pad_token = tokenizer.eos_token

# prompt = (
#     "Example:\n"
#     "Task: Make coffee. Step 1: Take a mug; Step 2: Add coffee; Step 3: Pour hot water; Step 4: Stir the coffee\n\n"
#     "Task: Make tea. Step 1: Pick up kettle; Step 2: Go to sink; Step 3: Fill kettle; Step 4: Turn on the stove;\n"
#     "Step 5:"
# )

# enc = tokenizer(prompt, return_tensors="pt").to(model.device)
# prompt_len = enc["input_ids"].shape[-1]

# stop_ids = []
# for s in [";", "\n"]:
#     stop_ids.extend(tokenizer.encode(s, add_special_tokens=False))

# stopping = StoppingCriteriaList([StopOnTokens(stop_ids, prompt_len)])

# with torch.no_grad():
#     out = model.generate(
#         **enc,
#         max_new_tokens=30,
#         do_sample=True,
#         temperature=0.7,
#         num_return_sequences=3,
#         top_p=0.95,
#         pad_token_id=tokenizer.pad_token_id,
#         eos_token_id=tokenizer.eos_token_id,
#         #stopping_criteria=stopping,
#     )

# print("--- Generated steps ---")
# for i in range(out.shape[0]):
#     gen_text = tokenizer.decode(out[i, prompt_len:], skip_special_tokens=True)
#     one_step = gen_text.split(";")[0].split("\n")[0].strip()
#     print(f"[{i}] {gen_text}")



# from src.model import HFLLMGenerate, GenConfig

# model_name = "meta-llama/Llama-2-7b-hf"
# tok = AutoTokenizer.from_pretrained(model_name)
# mdl = AutoModelForCausalLM.from_pretrained(model_name)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# mdl = mdl.to(device)

# gen = HFLLMGenerate(mdl, tok, GenConfig(max_new_tokens=16, temperature=1.0, top_p=1.0, do_sample=True))

# prompts = [
#     "Example:\nTask: Make coffee. Step 1: Take a mug; Step 2: Add coffee; Step 3: Pour hot water; Step 4: Stir the coffee\n\nTask: Make tea. Step 1: Pick up kettle; Step 2: Go to sink; Step 3: Fill kettle; Step 4: Turn on the stove;\nStep 5:",
#     "Example:\nTask: Make coffee. Step 1: Take a mug; Step 2: Add coffee; Step 3: Pour hot water; Step 4: Stir the coffee\n\nTask: Make tea. Step 1: Pick up kettle; Step 2: Go to sink; Step 3: Fill kettle; Step 4:",
#     "Example:\nTask: Make coffee. Step 1: Take a mug; Step 2: Add coffee; Step 3: Pour hot water; Step 4: Stir the coffee\n\nTask: Make tea. Step 1: Pick up kettle; Step 2: Go to sink; Step 3:"
# ]
# n = 3
# out = gen(prompts, n)

# # 1) структура
# assert isinstance(out, list) and len(out) == len(prompts)
# assert all(isinstance(x, list) and len(x) == n for x in out)

# # 2) поля и простые инварианты
# for i in range(len(prompts)):
#     for j in range(n):
#         d = out[i][j]
#         assert set(d.keys()) == {"text", "V_G", "sum_logprob", "n_tokens"}
#         assert isinstance(d["text"], str)
#         assert isinstance(d["n_tokens"], int)

#         if d["n_tokens"] > 0:
#             # sum_logprob == V_G * n_tokens (с маленькой погрешностью float)
#             assert abs(d["sum_logprob"] - d["V_G"] * d["n_tokens"]) < 1e-3

# print("OK")
# print(out)


from typing import List, Tuple
import re
import numpy as np
from sentence_transformers import SentenceTransformer

EOA = "<|eoa|>"

def clean_generated(text: str) -> str:
    """Keep only the first action-like phrase; drop 'Step k:' tail etc."""
    t = str(text).strip()
    # take text before "Step X:" if present
    t = re.split(r"\bStep\s*\d+\s*:\s*", t, maxsplit=1)[0].strip()
    # keep first clause before ';' (often the actual action)
    t = t.split(";")[0].strip()
    return t

def parse_admissible_actions(raw: str) -> List[str]:
    """Parse actions from a raw block where each line may end with <|eoa|>."""
    actions = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # strip token(s)
        if line.endswith(EOA):
            line = line[: -len(EOA)].strip()
        # skip special tokens / placeholders
        if line in {"@@UNKNOWN@@", "<|sact|>", "<|eact|>"}:
            continue
        # sometimes lines are like "<|sact|><|eoa|>"
        if "<|sact|>" in line or "<|eact|>" in line:
            continue
        actions.append(line)
    return actions

def map_to_closest_action(
    query_text: str,
    admissible_actions: List[str],
    model: SentenceTransformer,
) -> Tuple[str, float, int]:
    """
    Returns (best_action, cosine_sim, index).
    Uses SBERT embeddings + cosine similarity.
    """
    q = query_text

    # Encode (normalize_embeddings=True gives unit vectors => dot == cosine)
    q_emb = model.encode([q], normalize_embeddings=True)
    a_emb = model.encode(admissible_actions, normalize_embeddings=True)

    sims = (q_emb @ a_emb.T).squeeze(0)  # shape [A]
    best_idx = int(np.argmax(sims))
    return admissible_actions[best_idx], float(sims[best_idx]), best_idx


# --- Example usage ------------------------------------------------------------

query = "pour pancake batter; Step 4:"

raw_actions = """@@UNKNOWN@@
stir mixture<|eoa|>
<|sact|><|eoa|>
<|eact|><|eoa|>
whisk mixture<|eoa|>
add sugar<|eoa|>
pour water<|eoa|>
pour milk<|eoa|>
pour egg<|eoa|>
dip bread in mixture<|eoa|>
put bread in pan<|eoa|>
pour mixture into pan<|eoa|>
add onion<|eoa|>
flip pancake<|eoa|>
flip bread<|eoa|>
flip steak<|eoa|>
cut steak<|eoa|>
pour alcohol<|eoa|>
assemble shelve<|eoa|>
add flour<|eoa|>
close lid<|eoa|>
cut shelve<|eoa|>
melt butter<|eoa|>
put steak on grill<|eoa|>
remove bread from pan<|eoa|>
season steak<|eoa|>
open lid<|eoa|>
take pancake from pan<|eoa|>
add curry leaves<|eoa|>
add fish<|eoa|>
pull out dipstick<|eoa|>
stir<|eoa|>
top toast<|eoa|>
insert dipstick<|eoa|>
pour mixture into cup<|eoa|>
take steak from grill<|eoa|>
pour oil<|eoa|>
pour jello powder<|eoa|>
add chili powder<|eoa|>
spread creme upon cake<|eoa|>
taste steak<|eoa|>
pour espresso<|eoa|>
remove cap<|eoa|>
mix ingredients<|eoa|>
pour lemon juice<|eoa|>
add vanilla extract<|eoa|>
jack up<|eoa|>
tight wheel<|eoa|>
wipe off dipstick<|eoa|>
raise jack<|eoa|>
add strawberries to cake<|eoa|>
attach shelve<|eoa|>
get things out<|eoa|>
jack down<|eoa|>
unscrew wheel<|eoa|>
screw wheel<|eoa|>
squeeze lemon<|eoa|>
close cap<|eoa|>
pour lemonade into glass<|eoa|>
withdraw wheel<|eoa|>
start loose<|eoa|>
pour sesame oil<|eoa|>
add rice<|eoa|>
add kimchi<|eoa|>
paint shelve<|eoa|>
add meat<|eoa|>
put wheel<|eoa|>
add whipped cream<|eoa|>
sand shelve<|eoa|>
put bananas into blender<|eoa|>
add lettuce<|eoa|>
add mustard seeds<|eoa|>
put funnel<|eoa|>
spread mixture<|eoa|>
cut lemon<|eoa|>
put dough into form<|eoa|>
add cheese<|eoa|>
pour juice<|eoa|>
add tomato<|eoa|>
pack cucumbers in jar<|eoa|>
add spices<|eoa|>
add butter<|eoa|>
add ham<|eoa|>
check temperature<|eoa|>
lower jack<|eoa|>
remove funnel<|eoa|>
cut banana<|eoa|>
add tortilla<|eoa|>
put meringue into oven<|eoa|>
seal jar<|eoa|>
top steak<|eoa|>
pour vinegar<|eoa|>
cut strawberries<|eoa|>
put mixture into bag<|eoa|>
add coffee<|eoa|>
add ice<|eoa|>
add salt<|eoa|>
add taco<|eoa|>
cut cucumber<|eoa|>
peel banana<|eoa|>
steam milk<|eoa|>
move steak on grill<|eoa|>
brake on<|eoa|>
press coffee<|eoa|>
put jar in water<|eoa|>
put vegetables in water<|eoa|>
put things back<|eoa|>
cut onion<|eoa|>
"""

actions = parse_admissible_actions(raw_actions)

# pick a SBERT model (fast + good baseline)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

best_action, sim, idx = map_to_closest_action(query, actions, model)

print("Query (cleaned):", clean_generated(query))
print("Best action    :", best_action)
print("Cosine sim     :", sim)
print("Index          :", idx)
