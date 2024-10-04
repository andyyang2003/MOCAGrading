import os
import pandas as pd

# Define the directory where your images are stored
image_dir = 'C:/Users/Andy/Desktop/moca training/clocks/'  # Change to your folder path

# Get all image files from the directory (you can filter by specific extensions if needed)
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Sort image files: first by length (shorter names first), then lexicographically
image_files.sort(key=lambda x: (len(x), x))

# Create an empty list to store the file names and placeholder values
data = []

# Create rows with filenames and placeholder values (for manual entry later)
for image in image_files:
    data.append([image, None, None, None])  # None placeholders for circularity, number positioning, clock hands

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data, columns=['filename', 'circularity', 'number_positioning', 'clock_hands'])

# Define the CSV output file path
csv_output_path = 'C:/Users/Andy/Desktop/moca training/clocks/labels.csv'  # Adjust the path as needed

# Save the DataFrame to a CSV file
df.to_csv(csv_output_path, index=False)

print(f"CSV file created with {len(image_files)} images, sorted by filename length and lexicographically.")
