from pathlib import Path

# Categorical Columns
CATEGORICAL_COLS = ['Pclass', 'Sex', 'Embarked']

# [2] means go up 2 levels from the current file -> from ./04-pipeline-and-orchestration/lib//config.py to ./
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Path for dataset
DATA_DIRPATH = str(PROJECT_ROOT / "data")
# Path for models
MODELS_DIRPATH = str(PROJECT_ROOT / "models")
