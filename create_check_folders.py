import os

base_dir = r"c:\Users\udoo_w2\Desktop\work_traffic\matlab_multigpus"
print(f"Target Base Dir: {base_dir}")

if not os.path.exists(base_dir):
    print(f"CRITICAL: Base dir does not exist!")
    # Try to create it? But finding it means it should exist.
    exit(1)

folders = ["xFolder", "cFolder"]
for f in folders:
    path = os.path.join(base_dir, f)
    print(f"Attempting to create: {path}")
    try:
        os.makedirs(path, exist_ok=True)
        print(f"  os.makedirs returned without error.")
    except Exception as e:
        print(f"  Error creating {path}: {e}")

    if os.path.exists(path):
        print(f"  SUCCESS: {path} exists.")
        # Test write
        test_file = os.path.join(path, "test_write.txt")
        try:
            with open(test_file, "w") as tf:
                tf.write("test")
            print(f"  Write test successful: {test_file}")
        except Exception as e:
            print(f"  Write test FAILED: {test_file} - {e}")
    else:
        print(f"  FAILURE: {path} still does not exist after creation attempt.")

print("Listing matlab_multigpus content:")
try:
    print(os.listdir(base_dir))
except Exception as e:
    print(f"Error listing dir: {e}")
