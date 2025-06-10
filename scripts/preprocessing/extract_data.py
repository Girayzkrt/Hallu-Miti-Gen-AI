import os
import tarfile

download_dir = "pmc_downloads"
extracted_dir = "pmc_extracted"

os.makedirs(extracted_dir, exist_ok=True)

archives = [f for f in os.listdir(download_dir) if f.endswith(".tar.gz")]
total = len(archives)

for idx, filename in enumerate(archives, start=1):
    filepath = os.path.join(download_dir, filename)
    print(f"Extracting {filename}...")

    try:
        with tarfile.open(filepath, "r:gz") as tar:
            tar.extractall(path=extracted_dir)
    except Exception as e:
        print(f"Failed to extract {filename}: {e}")
    
    percent = (idx / total) * 100
    print(f"Progress: {percent:.1f}% ({idx}/{total})\n")

print("Done AQ!")
