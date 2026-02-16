import os

def count_lines(filepath):
    if not os.path.exists(filepath):
        return f"MISSING ({filepath})"
    with open(filepath, 'r', encoding='utf-8') as f:
        # Count lines. Note: Loop is memory efficient.
        return sum(1 for _ in f)

files = [
    'tr1_fix.csv', 
    'tr2_fix.csv', 
    'tr3_fix.csv', 
    '../tr_fix.csv'
]

output_lines = []
output_lines.append("--- LINE COUNTS ---")
total_split = 0
for f in files:
    count = count_lines(f)
    output_lines.append(f"{f}: {count}")
    
    # Validation Logic
    if isinstance(count, int) and 'tr_fix.csv' not in f:
        # Subtract header
        total_split += (count - 1)

output_lines.append("-" * 20)
output_lines.append(f"Sum of splits (images): {total_split}")

# Check total file
total_lines = count_lines('../tr_fix.csv')
if isinstance(total_lines, int):
    output_lines.append(f"../tr_fix.csv (Total file images): {total_lines - 1}")
    if total_split == (total_lines - 1):
        output_lines.append("VERIFICATION: MATCH")
    else:
        output_lines.append(f"VERIFICATION: MISMATCH (Diff: {total_split - (total_lines - 1)})")
else:
    output_lines.append("Could not verify total file.")

with open('counts_result.txt', 'w') as f:
    f.write('\n'.join(output_lines))
