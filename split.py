 
import os
import shutil
import random
from collections import defaultdict

# --- Configuration Variables (Modify these as needed) ---

# Path to your input dataset folder containing both images and label files
# Example: 'my_dataset_raw'
input_dataset_folder = 'Bangla_License_Plate'

# Path to the base folder where the split dataset will be created
# Example: 'my_dataset_split_yolo'
output_base_folder = 'Output'

# Define the proportions for train, validation, and test sets
# The sum of these proportions should be 1.0 (or close to it due to floating point arithmetic)
split_proportions = {
    'train': 0.7,
    'val': 0.2,
    'test': 0.1
}

# List of supported image file extensions (case-insensitive)
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# Label file extension (YOLO format)
label_extension = '.txt'

# --- End of Configuration Variables ---

print(f"Starting dataset splitting process...")
print(f"Input folder: {os.path.abspath(input_dataset_folder)}")
print(f"Output base folder: {os.path.abspath(output_base_folder)}")
print(f"Split proportions: {split_proportions}")

# 1. Collect all image and label file pairs
print("\nCollecting image and label file pairs...")
image_label_pairs = []
image_files_found = 0
label_files_found = 0
missing_labels = []
missing_images = []

for filename in os.listdir(input_dataset_folder):
    file_path = os.path.join(input_dataset_folder, filename)

    if os.path.isfile(file_path):
        base_name, ext = os.path.splitext(filename)
        ext = ext.lower()

        if ext in image_extensions:
            image_files_found += 1
            label_file = base_name + label_extension
            label_file_path = os.path.join(input_dataset_folder, label_file)

            if os.path.exists(label_file_path):
                image_label_pairs.append((file_path, label_file_path))
            else:
                missing_labels.append(filename)
        elif ext == label_extension:
            label_files_found += 1
            image_found = False
            for img_ext in image_extensions:
                if os.path.exists(os.path.join(input_dataset_folder, base_name + img_ext)):
                    image_found = True
                    break
            if not image_found:
                missing_images.append(filename)

if not image_label_pairs:
    print(f"Error: No matching image-label pairs found in '{input_dataset_folder}'.")
    print("Please ensure images (e.g., .jpg, .png) and corresponding .txt label files exist and have the same base name.")
    exit()

print(f"Found {len(image_label_pairs)} image-label pairs.")
if missing_labels:
    print(f"Warning: {len(missing_labels)} image files found without corresponding label files. They will be skipped.")
    # print(f"Skipped images: {missing_labels}") # Uncomment to see the list
if missing_images:
    print(f"Warning: {len(missing_images)} label files found without corresponding image files. They will be skipped.")
    # print(f"Skipped labels: {missing_images}") # Uncomment to see the list


# 2. Shuffle the pairs for random distribution
print("\nShuffling dataset...")
random.shuffle(image_label_pairs)

# 3. Calculate split sizes
total_samples = len(image_label_pairs)
split_counts = {}
current_idx = 0

for split_name, proportion in split_proportions.items():
    count = int(total_samples * proportion)
    split_counts[split_name] = count
    # Adjust last split to ensure all samples are included due to integer truncation
    if split_name == list(split_proportions.keys())[-1]:
        split_counts[split_name] = total_samples - current_idx
    current_idx += split_counts[split_name]

print(f"Total samples: {total_samples}")
print(f"Split distribution: {split_counts}")

# 4. Create output directories and copy files
print("\nCreating output directories and copying files...")
copied_counts = defaultdict(int)
current_sample_idx = 0

for split_name, count in split_counts.items():
    images_output_dir = os.path.join(output_base_folder, split_name, 'images')
    labels_output_dir = os.path.join(output_base_folder, split_name, 'labels')

    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)
    print(f"Created directories for '{split_name}':")
    print(f"  - {images_output_dir}")
    print(f"  - {labels_output_dir}")

    for i in range(count):
        if current_sample_idx >= total_samples:
            break # Safety break if somehow counts exceed total samples

        image_src_path, label_src_path = image_label_pairs[current_sample_idx]

        # Get just the filename (e.g., 'image1.jpg', 'image1.txt')
        image_filename = os.path.basename(image_src_path)
        label_filename = os.path.basename(label_src_path)

        # Define destination paths
        image_dest_path = os.path.join(images_output_dir, image_filename)
        label_dest_path = os.path.join(labels_output_dir, label_filename)

        # Copy files
        try:
            shutil.copy2(image_src_path, image_dest_path)
            shutil.copy2(label_src_path, label_dest_path)
            copied_counts[split_name] += 1
        except Exception as e:
            print(f"Error copying {image_filename} or {label_filename} to {split_name}: {e}")
        
        current_sample_idx += 1

print("\n--- Splitting Complete ---")
print("Summary of copied files:")
for split_name, count in copied_counts.items():
    print(f"- {split_name.capitalize()}: {count} image-label pairs")

print(f"\nDataset successfully split into '{output_base_folder}'")
print("You can now use this structure for YOLO model training.")