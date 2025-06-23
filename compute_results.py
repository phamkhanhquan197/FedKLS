import pandas as pd

threshold=0.89

def print_min_max_avg_table(csv_file_path):
    df = pd.read_csv(csv_file_path)
    # Select only numeric columns (excluding 'round' if you wish)
    numeric_cols = [col for col in df.columns if col != 'round' and pd.api.types.is_numeric_dtype(df[col])]
    summary = pd.DataFrame({
        "min": df[numeric_cols].min(),
        "max": df[numeric_cols].max(),
        "avg": df[numeric_cols].mean()
    })
    print(summary.T) 

def find_first_f1_threshold_round(csv_path: str, threshold: float = threshold) -> int:
    """Find first round where global F1-score meets/exceeds threshold."""
    try:
        df = pd.read_csv(csv_path)
        if 'local_f1' not in df.columns:
            print("Error: 'local_f1' column not found in CSV")
            return None
            
        for _, row in df.iterrows():
            if row['local_f1'] >= threshold:
                return int(row['round'])
        return None
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return None
    
def calculate_convergence_time(csv_path: str, convergence_round: int) -> float:
    """Calculate total processing time until (and including) convergence round."""
    try:
        df = pd.read_csv(csv_path)
        filtered_df = df[df['round'] <= convergence_round]
        total_time = filtered_df['processing_time'].sum()
        return total_time
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return None
    except KeyError:
        print("Error: 'processing_time' or 'round' column not found in CSV")
        return None

#Relative path to the CSV file
import os
csv_path = os.path.join(os.getcwd(), 'output', '2025-06-11', 'FedAvg', 'dirichlet_niid', 'lora_Banking77_1.0', 'FedAvg_legacy-datasets_banking77_dirichlet_niid_32_0.01_1.csv')
#Print only the name of method and dataset
print(f"CSV file path: {csv_path}")

print_min_max_avg_table(csv_path)
convergence_round = find_first_f1_threshold_round(csv_path)
print(f"Convergence round for F1-score >= {threshold}: {convergence_round}")
convergence_time = calculate_convergence_time(csv_path, convergence_round)
print(f"Total processing time until convergence (round {convergence_round}): {convergence_time:.4f} seconds = {convergence_time/60:.4f} minutes = {convergence_time/3600:.4f} hours")
