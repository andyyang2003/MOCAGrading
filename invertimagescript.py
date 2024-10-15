import cv2
import os

# Set the folder path
folder_path = 'clocks'

# List all files in the folder and filter for PNGs
files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

# Loop through the first 50 PNG files
for i, file_name in enumerate(files[:50]):
    # Construct the full file path
    file_path = os.path.join(folder_path, file_name)
    
    # Read the image
    image = cv2.imread(file_path)
    
    # Check if the image was loaded successfully
    if image is None:
        print(f"Error loading image {file_name}. Skipping...")
        continue
    
    # Invert the colors of the image
    inverted_image = cv2.bitwise_not(image)
    
    # Save the inverted image with a new name
    inverted_file_path = os.path.join(folder_path, f'inverted_{file_name}')
    cv2.imwrite(inverted_file_path, inverted_image)
    
    print(f"Processed and saved: {inverted_file_path}")

print("Color inversion complete for the first 50 images.")
