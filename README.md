# Grid Security AI

A deep learning framework for power grid security assessment using Graph Neural Networks (GNNs). The system simulates various grid contingencies (e.g., line or transformer outages), processes the simulation data, trains AI models to predict grid stability, and evaluates model performance.

## Features

- **Contingency Simulation**: Run power flow simulations for various grid failure scenarios using Pandapower
- **Graph Neural Networks**: Use GNNs to model complex grid topologies and predict security constraints
- **End-to-End Pipeline**: Complete workflow from simulation to evaluation with a single command
- **Multiple Grid Cases**: Support for different test cases (IEEE 1354, PEGASE 9241, etc.)
- **HPC Support**: Parallel processing capabilities for large-scale simulations

## Project Structure

```
├── config/           # Configuration files for different grid cases
├── data/             # Data directory (grids, simulations, processed datasets)
├── docs/             # Project documentation
├── models/           # Trained models and checkpoints
├── scripts/          # Scripts for running pipeline components
├── src/grid_ai/      # Main source code package
│   ├── simulation.py # Power grid simulation module
│   ├── preprocessing.py # Data preprocessing and graph construction
│   ├── model.py      # GNN model architecture
│   ├── train.py      # Training pipeline
│   └── evaluate.py   # Model evaluation and diagnostics
└── tests/            # Unit tests
```

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd grid-security-ai

# Create conda environment
conda env create -f environment.yml
conda activate grid_ai

# Install package in development mode
pip install -e .
```

### 2. Launch Web Interface (Recommended)

The easiest way to interact with Grid Security AI is through the web interface:

```bash
# Launch the Streamlit dashboard
streamlit run frontend/app.py

# Open your browser to http://localhost:8501
```

The web interface provides:
- **Dashboard**: System overview and quick pipeline execution
- **Run Pipeline**: Execute individual steps or full pipeline
- **View Results**: Browse evaluation metrics, plots, and training logs
- **Configuration**: Manage and view YAML configuration files
- **Logs**: Monitor system logs and pipeline execution

### 3. Command Line Usage

For command-line usage or automation:

```bash
# Run all steps with default fast configuration
python scripts/run_pipeline.py

# Or specify custom config
python scripts/run_pipeline.py --config config/pipeline.yml

# Run only specific steps
python scripts/run_pipeline.py --steps simulate preprocess
```

### 4. Manual Step-by-Step Execution

If you prefer to run individual components:

```bash
# 1. Run contingency simulations
python scripts/run_simulation_worker.py --config config/case_9241.yml

# 2. Preprocess simulation data into graph datasets
python scripts/run_preprocessing.py --config config/case_9241.yml

# 3. Train GNN model (fast mode for laptops)
python scripts/run_fast_training.py

# 4. Evaluate trained model
python scripts/run_evaluation.py --config config/case_9241.yml --model models/checkpoints_case_9241_fast/best_model.pt
```

## Configuration

### Pipeline Configuration

The `config/pipeline.yml` provides a unified configuration for the entire pipeline:

```yaml
pipeline:
  steps: ["simulate", "preprocess", "train", "evaluate"]
  case: "case_9241"
  mode: "fast"  # "fast" for laptop, "full" for production

case:
  name: "case_9241"
  grid_module: "pandapower.networks.case9241pegase"
  contingency_file: "data/contingencies/contingencies_9241.txt"
```

### Available Grid Cases

- **case_1354**: IEEE 1354-bus system
- **case_9241**: PEGASE 9241-bus system (default)

## Pipeline Details

### 1. Simulation
- Loads grid topology from Pandapower networks
- Applies contingencies (line/transformer outages)
- Runs AC/DC power flow simulations
- Saves results for all successful contingencies

### 2. Preprocessing
- Extracts node features (voltage, power injections)
- Constructs graph connectivity from grid topology
- Creates PyTorch Geometric Data objects
- Applies feature normalization

### 3. Training
- Trains GNN model on contingency datasets
- Supports GCN and GAT architectures
- Includes early stopping and gradient clipping
- Saves model checkpoints and training history

### 4. Evaluation
- Evaluates model on held-out test set
- Computes regression metrics (MAE, RMSE, R²)
- Generates prediction vs. ground truth plots
- Identifies worst-performing predictions

## Model Architecture

The GNN model uses:
- **Input**: Node features (voltage magnitude/angle, active/reactive power, contingency encoding)
- **Graph Structure**: Power grid connectivity (lines and transformers)
- **Architecture**: Configurable GNN layers (GCN/GAT) with residual connections
- **Output**: Graph-level predictions for voltage stability

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 1.12
- PyTorch Geometric
- Pandapower
- CUDA-compatible GPU (recommended for training)

## HPC Usage

For large-scale simulations on HPC clusters:

```bash
# Submit parallel simulation jobs
sbatch scripts/hpc/train_case_9241.slurm

# Or run specific tasks
python scripts/run_simulation_worker.py --config config/case_9241.yml --task-id 1
```

## Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=src/grid_ai tests/
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure PyTorch Geometric is properly installed
   ```bash
   pip install torch-geometric
   ```

2. **CUDA Errors**: Check GPU memory and CUDA version compatibility

3. **Missing Data**: Data directories are gitignored. Generate data using the pipeline or download sample datasets

4. **Memory Issues**: Use `mode: "fast"` in pipeline config for laptop usage

### Getting Help

- Check logs in `logs/` directory
- Review configuration files for parameter issues
- Ensure all dependencies are installed correctly

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Add license information]

## Citation

[Add citation information if applicable]
