AI Engine for Real-Time Power Grid Security Assessment

This project implements a complete, HPC-scalable pipeline for training and deploying a Graph Neural Network (GNN) to perform real-time security assessment on large-scale power grids.

The system is benchmarked on the PEGASE 9241-bus test case and is designed to predict post-contingency grid states (e.g., voltage violations, line overloads) orders of magnitude faster (>300x) than traditional physics-based solvers.

This repository contains the full workflow:

Data Generation: Scalable simulation of N-1 contingencies using pandapower and HPC (SLURM).

Data Preprocessing: Transformation of raw simulation results into graph-based datasets.

Model Training: A PyTorch-based GNN model optimized for grid topology.

Evaluation & Diagnostics: A suite of tools to analyze model performance against ground-truth simulations.

This work is intended as a production-grade asset for integration into real-time grid operations, digital twin platforms, and energy market analysis tools.

Project Structure

This project is organized as a professional Python package (src/grid_ai) with separate directories for orchestration scripts, tests, and configuration.

/src/grid_ai/: Core Python library for simulation, modeling, and training.

/scripts/: Runnable Python and shell scripts for orchestration (e.g., run_training.py).

/scripts/hpc/: SLURM job files for cluster-based data generation and training.

/scripts/launch/: High-level campaign launcher scripts.

/data/config/: Configuration files, such as contingency_list.txt.

/data/sample/: A small, self-contained data sample for running tests.

/notebooks/: Jupyter notebooks for data exploration and results analysis.

/tests/: Unit and integration tests for the grid_ai package.

/models/: Output directory for trained model checkpoints.

/docs/: Project documentation, including the commercial white paper.

1. Installation

Clone the repository:

git clone [https://github.com/your-username/grid_security_ai.git](https://github.com/your-username/grid_security_ai.git)
cd grid_security_ai


Create the Conda Environment:

conda env create -f environment.yml
conda activate grid_ai


Run Setup Script:
This script may install additional dependencies or configure the environment.

bash setup.sh


2. Quickstart & Testing

To verify the installation, run the test suite using a local data sample:

# (First, you may need to generate the local sample data)
python scripts/run_simulation.py --config config/local_test.yml

# Run tests
pytest


3. HPC Workflow

This project is designed to be run on an HPC cluster using SLURM.

Step 1: Generate Simulation Data

Modify the config and launch the data generation campaign. This will submit thousands of simulation jobs.

# This script intelligently manages the SLURM job array
bash scripts/launch/launch_campaign.sh


(This script would typically call sbatch scripts/hpc/generate_data.slurm)

Step 2: Train the Model

Once data generation is complete, preprocess the data and train the GNN.

# This script submits the GPU-enabled training job
sbatch scripts/hpc/train_model.slurm


Step 3: Run Diagnostics

Evaluate the trained model against a hold-out test set of contingencies.

# This script runs the evaluation and saves diagnostic plots/reports
sbatch scripts/hpc/run_diagnostics.slurm


4. Local Development

You can run the core components locally for development and debugging.

Run a single simulation:

python scripts/run_simulation.py --contingency "line_100_101"


Run training locally on sample data:

python scripts/run_training.py --data_path ./data/sample/ --epochs 50 --batch_size 32


Run evaluation locally:

python scripts/run_evaluation.py --model_path ./models/best_model.pt --data_path ./data/sample/
