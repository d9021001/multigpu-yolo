import os

target_dir = r"c:\Users\udoo_w2\Desktop\work_traffic\matlab_multigpus"
folders = ["xFolder", "cFolder"]

print(f"Checking {target_dir}...")
if not os.path.exists(target_dir):
    print(f"Error: {target_dir} does not exist!")
    exit(1)

for folder in folders:
    path = os.path.join(target_dir, folder)
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Created: {path}")
    except Exception as e:
        print(f"Failed to create {path}: {e}")

# Verify
import glob
print("Verification:")
print(glob.glob(os.path.join(target_dir, "*Folder")))
