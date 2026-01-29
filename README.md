# RLproject_PrioritizedLevelReplay
RL project
## Install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Train (PPO + PLR)
python src/train_plr_procgen.py

## Evaluate (generalization su livelli non visti)
python src/eval_procgen.py --ckpt runs/<RUN_NAME>/final.pt
