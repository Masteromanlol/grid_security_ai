# Grid Security AI

A deep learning approach to power grid security assessment.

## Project Structure

- `config/` - Configuration files for different grid cases
- `data/` - Data directory containing grid files and simulation results
- `docs/` - Project documentation
- `models/` - Trained models
- `scripts/` - Scripts for running simulations, training, and evaluation
- `src/` - Source code for the grid_ai package
- `tests/` - Unit tests

## Setup

1. Create conda environment:
   ```bash
   conda env create -f environment.yml
   ```

2. Activate environment:
   ```bash
   conda activate grid-security-ai
   ```

3. Install package in development mode:
   ```bash
   pip install -e .
   ```

## Usage

1. Run simulations:
   ```bash
   python scripts/run_simulation_worker.py --config config/case_1354.yml
   ```

2. Preprocess data:
   ```bash
   python scripts/run_preprocessing.py --config config/case_1354.yml
   ```

3. Train model:
   ```bash
   python scripts/run_training.py --config config/case_1354.yml
   ```

4. Evaluate model:
   ```bash
   python scripts/run_evaluation.py --config config/case_1354.yml --model models/model.pt
   ```

## Testing

Run tests:
```bash
pytest tests/
```