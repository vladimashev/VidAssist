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

from src.model import HFLLMGenerate, GenConfig

model_name = "meta-llama/Llama-2-7b-hf"
tok = AutoTokenizer.from_pretrained(model_name)
mdl = AutoModelForCausalLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
mdl = mdl.to(device)

gen = HFLLMGenerate(mdl, tok, GenConfig(max_new_tokens=16, temperature=1.0, top_p=1.0, do_sample=True))

prompts = [
    "Example:\nTask: Make coffee. Step 1: Take a mug; Step 2: Add coffee; Step 3: Pour hot water; Step 4: Stir the coffee\n\nTask: Make tea. Step 1: Pick up kettle; Step 2: Go to sink; Step 3: Fill kettle; Step 4: Turn on the stove;\nStep 5:",
    "Example:\nTask: Make coffee. Step 1: Take a mug; Step 2: Add coffee; Step 3: Pour hot water; Step 4: Stir the coffee\n\nTask: Make tea. Step 1: Pick up kettle; Step 2: Go to sink; Step 3: Fill kettle; Step 4:",
    "Example:\nTask: Make coffee. Step 1: Take a mug; Step 2: Add coffee; Step 3: Pour hot water; Step 4: Stir the coffee\n\nTask: Make tea. Step 1: Pick up kettle; Step 2: Go to sink; Step 3:"
]
n = 3
out = gen(prompts, n)

# 1) структура
assert isinstance(out, list) and len(out) == len(prompts)
assert all(isinstance(x, list) and len(x) == n for x in out)

# 2) поля и простые инварианты
for i in range(len(prompts)):
    for j in range(n):
        d = out[i][j]
        assert set(d.keys()) == {"text", "V_G", "sum_logprob", "n_tokens"}
        assert isinstance(d["text"], str)
        assert isinstance(d["n_tokens"], int)

        if d["n_tokens"] > 0:
            # sum_logprob == V_G * n_tokens (с маленькой погрешностью float)
            assert abs(d["sum_logprob"] - d["V_G"] * d["n_tokens"]) < 1e-3

print("OK")
print(out)