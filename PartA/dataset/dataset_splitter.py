import os
import random
import shutil
from tqdm import tqdm

def split_train_val(train_dir: str, val_dir: str, split_ratio: float = 0.2, seed: int = 42):
    random.seed(seed)

    # Make sure val_dir exists
    os.makedirs(val_dir, exist_ok = True)

    # Loop over each class folder
    for class_name in os.listdir(train_dir):
        class_train_path = os.path.join(train_dir, class_name)
        class_val_path = os.path.join(val_dir, class_name)

        if not os.path.isdir(class_train_path):
            continue

        # Create corresponding val subdirectory
        os.makedirs(class_val_path, exist_ok=True)

        # List all image files in the class
        all_images = os.listdir(class_train_path)
        total_images = len(all_images)
        num_val_images = int(split_ratio * total_images)

        # Randomly sample 20% of the images for val
        val_images = random.sample(all_images, num_val_images)

        # Move selected images to val folder
        for img_name in tqdm(val_images, desc=f"Moving {class_name} images"):
            src = os.path.join(class_train_path, img_name)
            dst = os.path.join(class_val_path, img_name)
            shutil.move(src, dst)

    print("\nValidation set created successfully.")

if __name__ == "__main__":
    dataset_root = "inaturalist_12K"
    train_path = os.path.join(dataset_root, "train")
    val_path = os.path.join(dataset_root, "val")

    split_train_val(train_path, val_path)