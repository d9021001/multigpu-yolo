import os
import math

def find_file(filename):
    if os.path.exists(filename):
        return filename
    parent = os.path.join('..', filename)
    if os.path.exists(parent):
        return parent
    return None

def split_csv_sequential():
    source_file = 'tr_fix.csv'
    found_path = find_file(source_file)
    
    if not found_path:
        print(f"❌ Error: Could not find '{source_file}' in current or parent directory.")
        return

    print(f"📖 Reading from: {found_path}")
    
    with open(found_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if not lines:
        print("❌ Error: File is empty.")
        return

    header = lines[0]
    data = lines[1:]
    total_rows = len(data)
    
    print(f"📊 Total Data Rows: {total_rows}")

    # Determine split sizes (Sequential, NO Randomization)
    # Split as evenly as possible. Remainder goes to the last file.
    split_size = total_rows // 3
    
    # Slice the data sequentially
    # Part 1: 0 -> split_size
    # Part 2: split_size -> 2*split_size
    # Part 3: 2*split_size -> end
    
    data1 = data[:split_size]
    data2 = data[split_size : 2*split_size]
    data3 = data[2*split_size :]
    
    len1, len2, len3 = len(data1), len(data2), len(data3)
    
    print("-" * 30)
    print(f"📝 Splitting into 3 files (Sequential)...")
    print(f"   tr1_fix.csv: {len1} rows")
    print(f"   tr2_fix.csv: {len2} rows")
    print(f"   tr3_fix.csv: {len3} rows")
    print("-" * 30)

    # Verification
    sum_splits = len1 + len2 + len3
    print(f"🔍 Verification:")
    print(f"   Sum of splits: {sum_splits}")
    print(f"   Original Total: {total_rows}")
    
    if sum_splits == total_rows:
        print("✅ SUCCESS: Counts Match Exactly.")
    else:
        print(f"❌ FAILURE: Mismatch detected! Diff: {sum_splits - total_rows}")
        return

    # Write files (Overwriting old ones)
    def write_file(name, rows):
        with open(name, 'w', encoding='utf-8') as f:
            f.write(header)
            f.writelines(rows)
        print(f"💾 Saved: {name}")

    write_file('tr1_fix.csv', data1)
    write_file('tr2_fix.csv', data2)
    write_file('tr3_fix.csv', data3)
    
    print("\nDone. You can verify the files now.")

if __name__ == "__main__":
    split_csv_sequential()
