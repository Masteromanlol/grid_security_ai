#!/usr/bin/env python
# aggregate_results.py
import os
import pickle
import pandas as pd
import argparse
from tqdm import tqdm

def aggregate_results(results_dir, output_file):
    """
    Aggregates individual simulation results from a directory into a single
    DataFrame and saves it as a pickle file.

    Args:
        results_dir (str): The directory containing the individual result files.
        output_file (str): The path to save the aggregated DataFrame.
    """
    all_results = []
    for filename in tqdm(os.listdir(results_dir)):
        if filename.endswith(".pkl"):
            file_path = os.path.join(results_dir, filename)
            with open(file_path, 'rb') as f:
                results = pickle.load(f)
                all_results.extend(results)

    df = pd.DataFrame(all_results)
    df.to_pickle(output_file)
    print(f"Aggregated {len(df)} results and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate simulation results.")
    parser.add_argument("results_dir", type=str, help="Directory containing the result files.")
    parser.add_argument("output_file", type=str, help="Path to save the aggregated DataFrame.")
    args = parser.parse_args()

    aggregate_results(args.results_dir, args.output_file)
