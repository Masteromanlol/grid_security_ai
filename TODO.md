# TODO: Add Frontend to Grid Security AI

## Overview
Create a Streamlit-based web interface that connects to the existing backend pipeline for power grid security assessment.

## Steps to Complete

### 1. Update Dependencies
- [x] Add Streamlit to `environment.yml`
- [x] Add Streamlit to `setup.py` install_requires

### 2. Create Frontend Structure
- [x] Create `frontend/` directory
- [x] Create `frontend/app.py` - Main Streamlit application
- [x] Create `frontend/utils.py` - Helper functions for backend connectivity

### 3. Implement Frontend Features
- [x] Dashboard page for pipeline overview
- [x] Pipeline execution page (run simulations, training, etc.)
- [x] Results visualization page (plots, metrics)
- [x] Configuration management page
- [x] Log viewing page

### 4. Connect Backend
- [x] Import grid_ai modules in utils.py
- [x] Implement functions to run pipeline scripts
- [x] Add error handling and status updates

### 5. Testing and Documentation
- [x] Update README.md with frontend usage instructions
- [x] Test Streamlit app locally
- [x] Verify backend connectivity (run simulations, view results)

### 6. Finalize
- [ ] Install updated dependencies
- [ ] Run full pipeline through frontend
- [ ] Address any issues or refinements
