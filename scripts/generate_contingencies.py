"""Generate contingencies for power grid."""

import os
import pandapower as pp
import pandapower.networks as pn

def generate_contingencies(net, output_file):
    """Generate N-1 contingencies for lines and transformers.
    
    Args:
        net: pandapower network
        output_file: path to output file
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    contingencies = []
    
    # Add line contingencies
    for idx, line in net.line.iterrows():
        if line.in_service:  # Only consider lines that are in service
            contingencies.append(f"line {idx}")
    
    # Add transformer contingencies
    for idx, trafo in net.trafo.iterrows():
        if trafo.in_service:  # Only consider transformers that are in service
            contingencies.append(f"transformer {idx}")
            
    # Write contingencies to file
    with open(output_file, 'w') as f:
        f.write("type,id\n")  # CSV header
        for cont in contingencies:
            component_type, component_id = cont.split()
            f.write(f"{component_type},{component_id}\n")
            
    print(f"Generated {len(contingencies)} contingencies:")
    print(f"- {sum(1 for c in contingencies if c.startswith('line'))} line contingencies")
    print(f"- {sum(1 for c in contingencies if c.startswith('transformer'))} transformer contingencies")
    
def main():
    # Load case1354pegase network
    net = pn.case1354pegase()
    
    # Generate contingencies
    output_file = "data/contingencies/contingencies_1354.txt"
    generate_contingencies(net, output_file)

if __name__ == "__main__":
    main()