import streamlit as st
import subprocess
import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Grid Security AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import grid_ai modules (optional, for direct function calls)
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))
try:
    from grid_ai import simulation, preprocessing, train, evaluate
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    st.warning("Backend modules not available. Using script execution only.")

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def run_pipeline_step(step_name, config_path):
    """Run a specific pipeline step."""
    try:
        if step_name == "simulate":
            script = "scripts/run_simulation_worker.py"
        elif step_name == "preprocess":
            script = "scripts/run_preprocessing.py"
        elif step_name == "train":
            script = "scripts/run_fast_training.py"
        elif step_name == "evaluate":
            script = "scripts/run_evaluation.py"
        else:
            raise ValueError(f"Unknown step: {step_name}")

        cmd = f"python {script} --config {config_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=Path(__file__).parent.parent)

        if result.returncode == 0:
            st.success(f"✅ {step_name.capitalize()} completed successfully!")
            return result.stdout
        else:
            st.error(f"❌ {step_name.capitalize()} failed!")
            st.code(result.stderr)
            return None
    except Exception as e:
        st.error(f"Error running {step_name}: {str(e)}")
        return None

def main():
    st.title("⚡ Grid Security AI Dashboard")
    st.markdown("A deep learning framework for power grid security assessment using Graph Neural Networks (GNNs).")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Run Pipeline", "View Results", "Configuration", "Logs"])

    if page == "Dashboard":
        show_dashboard()

    elif page == "Run Pipeline":
        show_pipeline_runner()

    elif page == "View Results":
        show_results_viewer()

    elif page == "Configuration":
        show_config_manager()

    elif page == "Logs":
        show_logs_viewer()

def show_dashboard():
    st.header("Dashboard")
    st.markdown("Overview of the Grid Security AI system.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Grid Cases", "3", "Available: 1354, 9241, 13659")

    with col2:
        st.metric("Models Trained", "0", "No models yet")

    with col3:
        st.metric("Simulations Run", "0", "No simulations yet")

    with col4:
        st.metric("Evaluations", "0", "No evaluations yet")

    st.markdown("---")
    st.subheader("Quick Actions")

    if st.button("🚀 Run Full Pipeline (Fast Mode)"):
        with st.spinner("Running full pipeline..."):
            config_path = "config/pipeline.yml"
            run_pipeline_step("simulate", config_path)
            run_pipeline_step("preprocess", config_path)
            run_pipeline_step("train", config_path)
            run_pipeline_step("evaluate", config_path)
        st.success("Pipeline execution completed!")

    st.markdown("---")
    st.subheader("System Status")
    st.info("System ready. Select a page from the sidebar to get started.")

def show_pipeline_runner():
    st.header("Run Pipeline")
    st.markdown("Execute individual pipeline steps or run the full pipeline.")

    # Pipeline steps
    steps = ["simulate", "preprocess", "train", "evaluate"]

    # Configuration selection
    config_options = {
        "Case 9241 (Default)": "config/case_9241.yml",
        "Case 1354": "config/case_1354.yml",
        "Pipeline Config": "config/pipeline.yml"
    }

    selected_config = st.selectbox("Select Configuration", list(config_options.keys()))
    config_path = config_options[selected_config]

    st.markdown("---")

    # Individual step execution
    st.subheader("Run Individual Steps")

    cols = st.columns(len(steps))
    for i, step in enumerate(steps):
        with cols[i]:
            if st.button(f"Run {step.capitalize()}", key=f"btn_{step}"):
                with st.spinner(f"Running {step}..."):
                    output = run_pipeline_step(step, config_path)
                    if output:
                        st.code(output[:500] + "..." if len(output) > 500 else output)

    st.markdown("---")

    # Full pipeline execution
    st.subheader("Run Full Pipeline")
    if st.button("🚀 Run All Steps", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, step in enumerate(steps):
            status_text.text(f"Running {step}...")
            output = run_pipeline_step(step, config_path)
            progress_bar.progress((i + 1) / len(steps))

        progress_bar.empty()
        status_text.empty()
        st.success("All pipeline steps completed!")

def show_results_viewer():
    st.header("View Results")
    st.markdown("View training results, evaluation metrics, and visualizations.")

    # Check for results
    results_dir = Path("results")
    plots_dir = Path("plots")

    if not results_dir.exists():
        st.warning("No results directory found. Run the pipeline first.")
        return

    # Evaluation results
    eval_file = results_dir / "evaluation" / "evaluation_results.pt"
    if eval_file.exists():
        st.subheader("Evaluation Results")
        try:
            import torch
            results = torch.load(eval_file, map_location='cpu')
            # Convert tensors to Python types for JSON display
            if isinstance(results, dict):
                display_results = {}
                for k, v in results.items():
                    if hasattr(v, 'item'):  # tensor
                        display_results[k] = v.item()
                    elif isinstance(v, list):
                        display_results[k] = [x.item() if hasattr(x, 'item') else x for x in v]
                    else:
                        display_results[k] = v
                st.json(display_results)
            else:
                st.write("Results:", results)
        except Exception as e:
            st.error(f"Error loading evaluation results: {e}")

    # Plots
    if plots_dir.exists():
        st.subheader("Diagnostic Plots")
        plot_files = list(plots_dir.glob("diagnostics/*.png"))
        if plot_files:
            for plot_file in plot_files:
                st.image(str(plot_file), caption=plot_file.name, use_column_width=True)
        else:
            st.info("No diagnostic plots available.")

    # Training history
    logs_dir = Path("logs")
    if logs_dir.exists():
        st.subheader("Training Logs")
        training_logs = list(logs_dir.glob("training*/*.log"))
        if training_logs:
            selected_log = st.selectbox("Select Training Log", [str(log) for log in training_logs])
            if selected_log:
                with open(selected_log, 'r') as f:
                    log_content = f.read()
                st.code(log_content, language="text")
        else:
            st.info("No training logs available.")

def show_config_manager():
    st.header("Configuration Manager")
    st.markdown("View and edit configuration files.")

    config_files = list(Path("config").glob("*.yml"))

    if not config_files:
        st.warning("No configuration files found.")
        return

    selected_config = st.selectbox("Select Configuration File", [str(f) for f in config_files])

    if selected_config:
        with open(selected_config, 'r') as f:
            config_content = f.read()

        st.subheader(f"Configuration: {selected_config}")
        st.code(config_content, language="yaml")

        # Display parsed config
        try:
            config_data = yaml.safe_load(config_content)
            st.subheader("Parsed Configuration")
            st.json(config_data)
        except Exception as e:
            st.error(f"Error parsing YAML: {e}")

def show_logs_viewer():
    st.header("Logs Viewer")
    st.markdown("View system logs and pipeline execution logs.")

    logs_dir = Path("logs")

    if not logs_dir.exists():
        st.warning("No logs directory found.")
        return

    # Get all log files
    log_files = list(logs_dir.rglob("*.log"))

    if not log_files:
        st.info("No log files found.")
        return

    selected_log = st.selectbox("Select Log File", [str(f.relative_to(logs_dir)) for f in log_files])

    if selected_log:
        log_path = logs_dir / selected_log
        with open(log_path, 'r') as f:
            log_content = f.read()

        st.subheader(f"Log: {selected_log}")
        st.code(log_content, language="text")

        # Log statistics
        lines = log_content.split('\n')
        st.markdown(f"**Total Lines:** {len(lines)}")
        errors = sum(1 for line in lines if 'ERROR' in line.upper())
        warnings = sum(1 for line in lines if 'WARNING' in line.upper())
        st.markdown(f"**Errors:** {errors} | **Warnings:** {warnings}")

if __name__ == "__main__":
    main()
