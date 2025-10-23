"""Test cases for simulation module."""

import unittest
import pandapower as pp
import pandas as pd
import numpy as np
from grid_ai.simulation import run_single_contingency, check_security_constraints

class TestSimulation(unittest.TestCase):
    def setUp(self):
        """Create a simple test network."""
        self.net = pp.create_empty_network()
        
        # Create buses
        b1 = pp.create_bus(self.net, vn_kv=110)
        b2 = pp.create_bus(self.net, vn_kv=110)
        b3 = pp.create_bus(self.net, vn_kv=110)
        
        # Create generator
        pp.create_gen(self.net, b1, p_mw=100, vm_pu=1.0)
        
        # Create loads
        pp.create_load(self.net, b2, p_mw=50, q_mvar=10)
        pp.create_load(self.net, b3, p_mw=50, q_mvar=10)
        
        # Create lines
        l1 = pp.create_line(self.net, b1, b2, length_km=10, std_type="NAYY 4x50 SE")
        l2 = pp.create_line(self.net, b2, b3, length_km=10, std_type="NAYY 4x50 SE")
        
        # Run initial power flow
        pp.runpp(self.net)
    
    def test_run_single_contingency(self):
        """Test single line contingency."""
        # Test line contingency
        result = run_single_contingency(self.net, {
            'type': 'line',
            'id': 0  # First line
        })
        
        self.assertTrue(result['success'])
        self.assertIn('bus_results', result)
        self.assertIn('line_results', result)
        self.assertIn('trafo_results', result)
        
        # Check that removing the only path to load causes convergence failure
        result = run_single_contingency(self.net, {
            'type': 'line',
            'id': 1  # Second line
        })
        self.assertTrue(result['success'])  # Should still converge due to first line
        
        # Test invalid contingency type
        with self.assertRaises(ValueError):
            run_single_contingency(self.net, {
                'type': 'invalid',
                'id': 0
            })
    
    def test_check_security_constraints(self):
        """Test security constraint checking."""
        # Create a normal state
        net = pp.create_empty_network()
        bus = pp.create_bus(net, vn_kv=110)
        pp.create_gen(net, bus, p_mw=0, vm_pu=1.0)
        pp.runpp(net)
        
        state = {
            'bus_results': pd.DataFrame({
                'vm_pu': [1.0],
                'va_degree': [0.0]
            }),
            'line_results': pd.DataFrame({
                'loading_percent': [50.0]
            }),
            'trafo_results': pd.DataFrame({
                'loading_percent': [60.0]
            })
        }
        
        checks = check_security_constraints(state)
        self.assertTrue(checks['voltage_in_limits'])
        self.assertTrue(checks['lines_not_overloaded'])
        self.assertTrue(checks['trafos_not_overloaded'])
        
        # Test violated constraints
        state['bus_results']['vm_pu'] = [0.94]  # Below 0.95
        state['line_results']['loading_percent'] = [110.0]  # Above 100%
        
        checks = check_security_constraints(state)
        self.assertFalse(checks['voltage_in_limits'])
        self.assertFalse(checks['lines_not_overloaded'])
        self.assertTrue(checks['trafos_not_overloaded'])