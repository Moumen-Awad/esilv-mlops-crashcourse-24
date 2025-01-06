from pathlib import Path

CATEGORICAL_COLS = ['Pclass', 'Sex', 'Embarked']

# [3] means go up 3 levels from the current file -> from ./04-pipeline-and-orchestration/lib//config.py to ./
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRPATH = str(PROJECT_ROOT / "data")
MODELS_DIRPATH = str(PROJECT_ROOT / "models")
