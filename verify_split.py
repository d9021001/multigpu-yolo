
import os

def count_lines(filepath):
    # Try current directory first
    if os.path.exists(filepath):
        print(f"[FOUND] {filepath}")
        with open(filepath, 'rb') as f:
            # Efficiently count lines and subtract header
            return sum(1 for _ in f) - 1
            
    # Try parent directory relative to current
    parent_path = os.path.join('..', filepath)
    if os.path.exists(parent_path):
        print(f"[FOUND (Parent)] {parent_path}")
        with open(parent_path, 'rb') as f:
            return sum(1 for _ in f) - 1
            
    print(f"[MISSING] Cannot find: {filepath}")
    return 0

def scan_files():
    print("-" * 30)
    print("VERIFYING CSV SPLIT CONSISTENCY")
    print("-" * 30)
    
    # 1. Count individual splits
    n1 = count_lines('tr1_fix.csv')
    print(f"  > tr1_fix: {n1} images")
    
    n2 = count_lines('tr2_fix.csv')
    print(f"  > tr2_fix: {n2} images")
    
    n3 = count_lines('tr3_fix.csv')
    print(f"  > tr3_fix: {n3} images")
    
    current_sum = n1 + n2 + n3
    print("-" * 30)
    print(f"SUM OF SPLITS: {current_sum}")
    
    # 2. Count total file (try both locations)
    if os.path.exists('tr_fix.csv'):
        n_total = count_lines('tr_fix.csv')
    else:
        # Check parent ..
        n_total = count_lines('tr_fix.csv') # Function handles parent look-up
        
    print(f"TOTAL FILE:    {n_total}")
    print("-" * 30)
    
    # 3. Compare with strict equality
    if current_sum == n_total:
        print("✅ MATCH: The split files sum perfectly to the total file.")
    else:
        diff = current_sum - n_total
        print(f"❌ MISMATCH: Difference of {diff} images.")
        if diff > 0:
            print("   -> Splits contain DUPLICATES (More than total).")
        else:
            print("   -> Splits are MISSING data (Less than total).")
            
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    scan_files()
