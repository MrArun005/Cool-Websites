# bolt-coder: our own small coding model

Qwen2.5-Coder-0.5B-Instruct fine-tuned with LoRA on this repository (code, docs, commit history), trained on CPU.

| step | command | time on 4 CPU cores |
|---|---|---|
| data | `python coder/prepare_data.py` → `coder/data/` | 30 s |
| train | `python3 tools/keepalive/keepalive.py start coder/jobs/c1.json` (background, resumes from checkpoints) | ~4 h (≈115 tokens/s) |
| exam | `coder/jobs/c1-exam.json` runs `coder/exam.py` on the base model and the adapter once training is done | ~1 h |
| self-improve | `coder/jobs/si.json` runs `coder/self_improve.py` after the exam: practice on MBPP, keep only answers that pass their tests, retrain, re-exam, keep the round only if better | ~3–4 h per round |
| package | `python coder/package.py --run c1 --name bolt-coder-0.5B` → `models/bolt-coder-0.5B/` | 1 min |

Setup: `pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install transformers peft accelerate pyarrow`.
The job files point at the session's venv python; edit `cmd` if yours is elsewhere.
On a GPU the same `train_lora.py` uses CUDA + bf16 automatically (try `--base Qwen/Qwen2.5-Coder-1.5B-Instruct`).
