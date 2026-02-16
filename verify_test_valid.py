
import os
import csv

def verify_csv_images(filename):
    count = 0
    filepath = filename
    
    # Try current or parent directory
    if not os.path.exists(filepath):
        parent = os.path.join('..', filename)
        if os.path.exists(parent):
            filepath = parent
        else:
            print(f"❌ [MISSING] {filename}")
            return

    # Count lines (excluding header)
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            headers = next(reader) # skip header
            for row in reader:
                if row: # ignore empty lines
                    count += 1
        except StopIteration:
            pass # empty file
    
    print(f"✅ {filename}: {count} images")

print("-" * 30)
print("VERIFYING IMAGE COUNTS")
print("-" * 30)

verify_csv_images('test1.csv')
verify_csv_images('valid1_fix.csv')

print("-" * 30)
input("\nPress Enter to exit...")
