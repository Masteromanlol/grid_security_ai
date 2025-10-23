import os, pickle
import pandas as pd
import numpy as np
os.makedirs('data/raw/simulation_results', exist_ok=True)
# Build a tiny synthetic simulation result matching expectations
n = 1354
bus_res = pd.DataFrame({
    'vm_pu': np.ones(n),
    'va_degree': np.zeros(n),
    'p_mw': np.zeros(n),
    'q_mvar': np.zeros(n)
})
line_res = pd.DataFrame({
    'p_from_mw': np.zeros(max(1, int(n*1.5))),
    'q_from_mvar': np.zeros(max(1, int(n*1.5))),
    'pl_mw': np.zeros(max(1, int(n*1.5))),
    'loading_percent': np.zeros(max(1, int(n*1.5))),
    'max_i_ka': np.ones(max(1, int(n*1.5)))
})
trafo_res = pd.DataFrame({'loading_percent': np.zeros(1)})
result = {
    'success': True,
    'bus_results': bus_res,
    'line_results': line_res,
    'trafo_results': trafo_res,
    'contingency': {'type': 'line', 'id': 1}
}
with open('data/raw/simulation_results/sample_0.pkl','wb') as f:
    pickle.dump(result, f)
print('Wrote synthetic sample to data/raw/simulation_results/sample_0.pkl')
