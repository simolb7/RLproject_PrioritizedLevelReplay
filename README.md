# RLproject_PrioritizedLevelReplay
RL project

## Install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Train Baseline
python src/train_baseline.py --env <GAME_NAME> --config configs/<GAME_NAME>.yaml

with <GAME_NAME> between ["coinrun", "bigfish", "chaser", "starpilot", "jumper"]


## Train with PLR
python src/train_plr_procgen.py --env <GAME_NAME> --config configs/<GAME_NAME>.yaml

with <GAME_NAME> between ["coinrun", "bigfish", "chaser", "starpilot", "jumper"]


## Evaluate Baseline or PLR
python src/eval_procgen.py --env <GAME_NAME> --config configs/<GAME_NAME>.yaml --ckpt runs/<RUN_NAME>/final.pt --render --level <SEED_LEVEL> --save-gif <SAVE_PATH>

with <GAME_NAME> between ["coinrun", "bigfish", "chaser", "starpilot", "jumper"], --render if you want to see the render of the evaluation, --level <SEED_LEVEL> to evaluate a single level, --save-gif <SAVE_PATH> to save the render as a gif in a specific folder
