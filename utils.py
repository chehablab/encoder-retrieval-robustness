import subprocess
import os, re, sys
from tqdm.notebook import tqdm

def download_using_axel(url, output_dir, output_filename, num_connections = 10):
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if already_downloaded(output_dir, output_filename):
        print(f"File {output_filename} already exists and appears to be completely downloaded. Skipping download.")
        return

    if not axel_available():
        raise Exception("Axel is not installed. Please run install_axel.sh first.")


    print(f"Downloading {url} using Axel with {num_connections} connections...")

    process = subprocess.Popen(
        ["axel",
            "-n", str(num_connections),
            "-o", os.path.join(output_dir, output_filename),
            url],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

    pbar = tqdm(total=100, desc="Downloading", unit="%")

    for line in process.stdout:
        line = line.decode('utf-8')
        match = re.search(r'(\d{1,3})%', line)
        if match:
            progress = int(match.group(1))
            pbar.n = progress
            pbar.refresh()

    process.wait()
    pbar.close()

    if process.returncode != 0:
        raise Exception(f"Axel failed with return code {process.returncode}")

    print("Download completed successfully!")

def axel_available():
    result = subprocess.run(["axel", "--version"], 
                            capture_output=True,
                            text=True,
                            check=False)
    if result.returncode != 0: 
        return False
    else:
        return True
        
def already_downloaded(output_dir, output_filename):
    output_path = os.path.join(output_dir, output_filename)
    partial_file = output_path + ".st"
    if os.path.exists(output_path) and not os.path.exists(partial_file):
        return True
    else:
        return False

import random
from collections import defaultdict

def stratified_sample(dataset, sample_size):
    class_indices = defaultdict(list)
    for i, (_, label) in tqdm(enumerate(dataset), total=len(dataset), desc="Indexing classes"):
        class_indices[label].append(i)
    total_size = len(dataset)
    sampled_indices = []
    for label, indices in class_indices.items():
        n_samples = max(
            1,
            round(sample_size * len(indices) / total_size)
        )
        sampled_indices.extend(
            random.sample(indices, min(n_samples, len(indices)))
        )
    sampled_indices = random.sample(
        sampled_indices,
        min(sample_size, len(sampled_indices))
    )
    return [dataset[i] for i in sampled_indices]
