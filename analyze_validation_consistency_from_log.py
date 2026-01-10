"""
Phân tích tính nhất quán của validation set từ log file.
Kiểm tra xem validation samples có thay đổi giữa các rounds khi không có dữ liệu mới.
"""

import re
from collections import defaultdict
from pathlib import Path

def parse_log_file(log_path: str):
    """
    Parse log file để extract validation và training samples cho mỗi client và round.
    """
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern để match validation samples
    val_pattern = r'Client (\d+) \(Total validation samples: (\d+)'
    # Pattern để match training samples
    train_pattern = r'Client (\d+) \(Total training samples: (\d+)'
    # Pattern để match round number
    round_pattern = r'======================================Round (\d+)======================================'
    
    # Extract rounds
    rounds = []
    for match in re.finditer(round_pattern, content):
        round_num = int(match.group(1))
        rounds.append(round_num)
    
    # Extract validation và training samples
    client_data = defaultdict(lambda: defaultdict(dict))  # client_id -> round -> {'train': int, 'val': int}
    
    current_round = None
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        # Check for round
        round_match = re.search(round_pattern, line)
        if round_match:
            current_round = int(round_match.group(1))
            continue
        
        # Check for validation samples
        val_match = re.search(val_pattern, line)
        if val_match and current_round:
            client_id = int(val_match.group(1))
            val_samples = int(val_match.group(2))
            client_data[client_id][current_round]['val'] = val_samples
        
        # Check for training samples
        train_match = re.search(train_pattern, line)
        if train_match and current_round:
            client_id = int(train_match.group(1))
            train_samples = int(train_match.group(2))
            client_data[client_id][current_round]['train'] = train_samples
    
    return client_data, rounds

def analyze_validation_consistency(log_path: str):
    """
    Phân tích tính nhất quán của validation set từ log.
    """
    print("=" * 80)
    print("Phân Tích Tính Nhất Quán Validation Set Từ Log")
    print("=" * 80)
    
    client_data, rounds = parse_log_file(log_path)
    
    print(f"\nTìm thấy {len(rounds)} rounds: {sorted(rounds)}")
    print(f"Tìm thấy {len(client_data)} clients: {sorted(client_data.keys())}")
    
    all_issues = []
    all_checks_passed = True
    
    for client_id in sorted(client_data.keys()):
        print(f"\n{'='*80}")
        print(f"Client {client_id}")
        print(f"{'='*80}")
        
        prev_val_samples = None
        prev_train_samples = None
        
        for round_num in sorted(rounds):
            if round_num not in client_data[client_id]:
                continue
            
            data = client_data[client_id][round_num]
            train_samples = data.get('train', None)
            val_samples = data.get('val', None)
            
            print(f"\n  Round {round_num}:")
            
            if train_samples is not None:
                print(f"    Training samples: {train_samples}")
            else:
                print(f"    Training samples: N/A")
            
            if val_samples is not None:
                print(f"    Validation samples: {val_samples}")
            else:
                print(f"    Validation samples: N/A")
            
            # Check 1: Validation set should not be empty
            if val_samples is not None and val_samples == 0:
                issue = f"Client {client_id}, Round {round_num}: Validation set is EMPTY!"
                all_issues.append(issue)
                all_checks_passed = False
                print(f"    ❌ ERROR: {issue}")
            
            # Check 2: If no new training data, validation should be same
            if round_num > 1 and prev_train_samples is not None and train_samples is not None:
                has_new_data = train_samples > prev_train_samples
                
                if not has_new_data:
                    # No new data, validation should be same
                    if prev_val_samples is not None and val_samples is not None:
                        if prev_val_samples == val_samples:
                            print(f"    ✓ Validation unchanged (no new data)")
                        else:
                            issue = f"Client {client_id}, Round {round_num}: Validation changed from {prev_val_samples} to {val_samples} without new data!"
                            all_issues.append(issue)
                            all_checks_passed = False
                            print(f"    ❌ ERROR: {issue}")
            
            # Check 3: If new training data, validation should accumulate (increase or stay same)
            if round_num > 1 and prev_train_samples is not None and train_samples is not None:
                has_new_data = train_samples > prev_train_samples
                
                if has_new_data:
                    if prev_val_samples is not None and val_samples is not None:
                        if val_samples >= prev_val_samples:
                            increase = val_samples - prev_val_samples
                            print(f"    ✓ Validation accumulated (+{increase} samples)")
                        else:
                            issue = f"Client {client_id}, Round {round_num}: Validation decreased from {prev_val_samples} to {val_samples} with new data!"
                            all_issues.append(issue)
                            all_checks_passed = False
                            print(f"    ❌ ERROR: {issue}")
            
            prev_val_samples = val_samples
            prev_train_samples = train_samples
    
    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if all_checks_passed:
        print("✓ ALL CHECKS PASSED!")
        print("  - Validation sets are consistent across rounds")
        print("  - Validation sets accumulate correctly when new data arrives")
        print("  - Validation sets remain unchanged when no new data")
    else:
        print("❌ ISSUES FOUND:")
        for issue in all_issues:
            print(f"  - {issue}")
    
    return all_checks_passed

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze validation consistency from log")
    parser.add_argument("--log", type=str, 
                        default="output/2026-01-10/FedAvg/dirichlet_niid/2/log.txt",
                        help="Path to log file")
    args = parser.parse_args()
    
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)
    
    success = analyze_validation_consistency(str(log_path))
    sys.exit(0 if success else 1)
