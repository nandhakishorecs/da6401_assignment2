import os
from PIL import Image           # type: ignore

def get_image_dimensions(folder_path):
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        return
    
    # Supported image formats
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
    
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if file is an image
        if filename.lower().endswith(image_extensions):
            file_path = os.path.join(folder_path, filename)
            try:
                # Open image and get dimensions
                with Image.open(file_path) as img:
                    width, height = img.size
                    print(f"{filename}: {width}x{height} pixels")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    # Specify the folder path here
    folder_path = "/Users/nandhakishorecs/Documents/IITM/Jan_2025/DA6401/Assignments/Assignment2/PartA/dataset/inaturalist_12K"
    get_image_dimensions(folder_path)