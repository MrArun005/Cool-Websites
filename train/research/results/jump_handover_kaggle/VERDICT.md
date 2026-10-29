# Kaggle jump-handover (29 Oct): better alone, worse inside MasterMind -> NOT used (models/bolt-dog-jump-handover-74M)

Held-out test hand-overs (100 each, brake on): 87/86/91/90% at 20/30/40/50 cm vs 84/90/87/73% for jump-handover 59M.
Kaggle's best/ stayed at the start (its 10-try in-training exam is too noisy to pick on); re-tested checkpoints 62, 66,
70, 74.18M on dev hand-overs and picked 74.18M. Model card has every number and the commands.

Inside the teacher chain (v9j vs v8g, dev seeds, 30 tries per switch) trot -> jump 50 cm fell 23 -> 14 of 30 and the
chain 94.3 -> 93.6%, so jump-handover 59M stays. Cause and next step: the model card.
