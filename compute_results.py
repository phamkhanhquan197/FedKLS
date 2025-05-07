import pandas as pd

def print_min_max_avg_table(csv_file_path):
    df = pd.read_csv(csv_file_path)
    # Select only numeric columns (excluding 'round' if you wish)
    numeric_cols = [col for col in df.columns if col != 'round' and pd.api.types.is_numeric_dtype(df[col])]
    summary = pd.DataFrame({
        "min": df[numeric_cols].min(),
        "max": df[numeric_cols].max(),
        "avg": df[numeric_cols].mean()
    })
    summary.index.name = ""
    print(summary.T) 

def find_first_f1_threshold_round(csv_path: str, threshold: float = 0.57) -> int:
    """Find first round where global F1-score meets/exceeds threshold."""
    try:
        df = pd.read_csv(csv_path)
        if 'global_f1_score' not in df.columns:
            print("Error: 'global_f1_score' column not found in CSV")
            return None
            
        for _, row in df.iterrows():
            if row['global_f1_score'] >= threshold:
                return int(row['round'])
        return None
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return None

print_min_max_avg_table('/home/quan/FedKL-SVD/FedEasy/out/2025-05-07/FedAvg/dirichlet_niid/FFT_alpha1.0/FedAvg_SetFit_20_newsgroups_dirichlet_niid_32_0.01_1_metrics.csv')
convergence_round = find_first_f1_threshold_round('/home/quan/FedKL-SVD/FedEasy/out/2025-05-07/FedAvg/dirichlet_niid/FFT_alpha1.0/FedAvg_SetFit_20_newsgroups_dirichlet_niid_32_0.01_1_metrics.csv', threshold=0.57)
print(f"Convergence round for F1-score >= 0.57: {convergence_round}")