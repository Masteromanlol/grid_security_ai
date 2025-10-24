"""Generate contingencies for power grid."""

import os
import argparse
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
    parser = argparse.ArgumentParser(description='Generate contingencies for power grid case')
    parser.add_argument('--case', type=str, required=True,
                       help='Grid case name (e.g., case1354pegase, case9241pegase)')
    args = parser.parse_args()

    # Load the specified network
    if args.case == 'case1354pegase':
        net = pn.case1354pegase()
        output_file = "data/contingencies/contingencies_1354.txt"
    elif args.case == 'case9241pegase':
        net = pn.case9241pegase()
        output_file = "data/contingencies/contingencies_9241.txt"
    else:
        raise ValueError(f"Unsupported case: {args.case}")

    # Generate contingencies
    generate_contingencies(net, output_file)

if __name__ == "__main__":
    main()
