# bolt-coder-si-0.5B

Qwen/Qwen2.5-Coder-0.5B-Instruct that improved itself on CPU, by expert iteration (coder/self_improve.py): each round
it tried 474 MBPP practice problems 4 times, a sandbox ran every try against the problem's tests, and it learned only
from tries that passed (up to 2 per problem), mixed with 300 examples of this repo's own code. A round is kept only if
HumanEval + MBPP exam passes do not drop and the repo held-out loss is not more than 2% worse.
This adapter is round 2.

## Rounds

Exams: HumanEval (164) and the first 200 MBPP test problems, never practiced on; greedy, pass@1. Repo loss = loss per
token on 104 held-out examples from files and commits never trained on (lower is better).

| round | HumanEval | MBPP | exam passes | repo loss | practice solved | verified solutions | |
|---|---|---|---|---|---|---|---|
| 0 | 56.7% | 28.0% | 149 | 2.926 | –/– | – | kept |
| 1 | 54.3% | 34.0% | 157 | 2.599 | 236/474 | 418 | kept |
| 2 | 51.2% | 36.5% | 157 | 2.470 | 263/474 | 464 | kept |

Round 0 is the plain base model. Every round's tries (passed or not), training data, train log and exam are in
round_<n>/.

## Use it

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
model = PeftModel.from_pretrained(model, "models/bolt-coder-si-0.5B/adapter").merge_and_unload()
msgs = [{"role": "user", "content": "Write a function that returns the n-th Fibonacci number."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
print(tok.decode(model.generate(ids, max_new_tokens=256)[0][ids.shape[1]:], skip_special_tokens=True))
```

For Ollama / llama.cpp: `model.save_pretrained("merged"); tok.save_pretrained("merged")`, convert with llama.cpp's
`convert_hf_to_gguf.py merged --outtype q8_0`, then `ollama create` from a Modelfile `FROM ./merged-q8_0.gguf`.

## Run it again

`python3 tools/keepalive/keepalive.py start coder/jobs/si.json` (resumes where it stopped), then
`python coder/package_si.py --name bolt-coder-si-0.5B`.
