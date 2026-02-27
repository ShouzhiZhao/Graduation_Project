import os
from pathlib import Path

# Base directory of the project
PROJECT_ROOT = Path(__file__).parent.absolute()

# Simulation Agent paths
SIMULATION_AGENT_DIR = PROJECT_ROOT / "simulation_agent"
SIMULATION_DATA_DIR = PROJECT_ROOT / "simulation_data"
RELATIONSHIP_DATA_DIR = SIMULATION_AGENT_DIR / "data"

# DeepIM paths
DEEPIM_DIR = PROJECT_ROOT / "DeepIM"
DEEPIM_CHECKPOINTS_DIR = DEEPIM_DIR / "checkpoints"

# Data Analyze paths
DATA_ANALYZE_DIR = PROJECT_ROOT / "real_data"

# External Models paths
# Assuming models are in a parallel directory to the original project or specific path
# For now, pointing to the known location.
MODELS_DIR = Path("/mlp_vepfs/share/zsz/models")

def get_relationship_file(filename="relationship_1000.csv"):
    return RELATIONSHIP_DATA_DIR / filename

def get_model_path(model_name):
    return MODELS_DIR / model_name

def get_deepim_checkpoint(filename):
    return DEEPIM_CHECKPOINTS_DIR / filename

def get_simulation_output_dir(target_movie=None):
    if target_movie is None:
        return SIMULATION_DATA_DIR
    return os.path.join(SIMULATION_DATA_DIR, target_movie.replace("/", "_").replace(" ", "_"))
    