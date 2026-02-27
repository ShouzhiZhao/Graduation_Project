# Graduation Project

This project unifies `simulation_agent`, `DeepIM`, and `data_analyze` into a single cohesive system.

## Project Structure

- `simulation_agent/`: Contains the simulation logic and data generator.
- `DeepIM/`: Contains the Deep Influence Maximization model (training and optimization).
- `real_data/`: Contains data analysis scripts.
- `main.py`: Unified entry point to run different components.
- `paths.py`: Centralized path management configuration.

## Usage

You can use `main.py` to run different stages of the project.

### 1. Run Simulation
Generate simulation data for a specific movie.
```bash
python main.py simulate --target_movie_id 0 --config_file config/config.yaml
```

### 2. Train DeepIM Model
Train the model using the generated data.
```bash
python main.py train --epochs 100
```

### 3. Run Optimization
Optimize seed selection using the trained model.
```bash
python main.py optimize --seed_num 10 --iterations 500
```

### 4. Run Analysis
Run the analysis scripts.
```bash
python main.py analyze
```

## Configuration

- `paths.py` contains absolute and relative paths for project components. Modify this file if you move directories or external models.
- `simulation_agent/config/config.yaml` controls the simulation parameters.
