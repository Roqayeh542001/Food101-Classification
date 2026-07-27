from torchvision.datasets import Food101

from pathlib import Path
import tarfile


DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "food-101"
ARCHIVE = DATA_DIR / "food-101.tar.gz"

def download_dataset():
    if DATASET_DIR.exists():
        print("Food101 already extracted.")

    print("Downloading Food101 ...")

    Food101(
        root=DATA_DIR,
        split="train",
        download=True
    )

    Food101(
        root=DATA_DIR,
        split="test",
        download=True
    )

    print("Download finished.")

def extract_dataset():
    if DATASET_DIR.exists():
        print("Dataset already extracted.")
        return
    
    if not ARCHIVE.exists():
        print("Archive not found.")
        return
    
    print("Extracting dataset ...")

    with tarfile.open(ARCHIVE, "r:gz") as tar:
        tar.extractall(DATA_DIR)

    print("Extraction finished.")

if __name__ == "__main__":
    #download_dataset()
    extract_dataset()
