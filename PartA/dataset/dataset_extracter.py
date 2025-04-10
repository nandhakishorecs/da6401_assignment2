import requests
from tqdm import tqdm
import os

def download_file(url: str, output_path: str):
    
    response = requests.get(url, stream = True)
    response.raise_for_status()  # Error logging

    total_size = int(response.headers.get('content-length', 0))
    filename = os.path.basename(output_path)

    with open(output_path, 'wb') as f, tqdm(
        desc = f"Downloading {filename}",
        total = total_size,
        unit = 'B',
        unit_scale = True,
        unit_divisor = 1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

if __name__ == "__main__":
    # Replace with your URL
    dataset_url = "https://storage.googleapis.com/wandb_datasets/nature_12K.zip"  
    
    # Destination filename
    output_path = "/Users/nandhakishorecs/Documents/IITM/Jan_2025/DA6401/Assignments/Assignment2/PartA/dataset/inaturalist_12K/dataset.zip"

    download_file(dataset_url, output_path)
