import os
import laspy
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def main(args):
    """
    Main function to process the LAS file.
    This version is flexible for feature and class count based on user input.
    """
    print(f"Starting data preparation with {args.num_features} features and {args.num_classes} classes...")

    # --- 1. Load LAS File ---
    try:
        las_file = laspy.read(args.file_path)
        print(f"Successfully loaded {args.file_path}")
    except Exception as e:
        print(f"Error loading LAS file: {e}")
        return

    # --- 2. Extract Coordinates, Features, and Labels ---
    points = np.vstack((las_file.x, las_file.y, las_file.z)).transpose()
    try:
        if args.num_features == 4:
            print("Extracting 4 features (X, Y, Z, Intensity)...")
            features = las_file.intensity.reshape(-1, 1)
        elif args.num_features == 7:
            print("Extracting 7 features (X, Y, Z, Intensity, Range, Ring, AngleOfIncidence)...")
            features = np.vstack((
                las_file.intensity,
                las_file.Range,
                las_file.Ring,
                las_file.AngleOfIncidence
            )).transpose()
        else:
            raise ValueError("num_features must be 4 or 7")

    except AttributeError as e:
        print(f"Feature extraction failed. Attribute not found: {e}")
        return

    # --- Label Mapping ---
    classification = np.array(las_file.classification)
    if args.num_classes == 7:
        print("Mapping to 7 classes (6 main + 1 'other')")
        labels = np.full(classification.shape, 6, dtype=np.int32) 
        for i in range(1, 7):
            labels[classification == i] = i - 1
    else: # Default to 6 classes
        print("Mapping to 6 classes (points not in 1-6 will be ignored during training)")
        labels = classification - 1
    
    print(f"Extracted {len(points)} points. Unique labels created: {np.unique(labels)}")

    # --- 3. Normalize Features ---
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    all_data = np.hstack((points, features_normalized))
    print("Features normalized.")

    # --- 4. Spatially Divide into Blocks with Overlap (Sliding Window) ---
    print(f"Dividing point cloud into blocks of size {args.block_size}x{args.block_size}m with {args.overlap*100}% overlap...")
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    
    stride = args.block_size * (1 - args.overlap)

    grid_x = np.arange(min_coords[0], max_coords[0], stride)
    grid_y = np.arange(min_coords[1], max_coords[1], stride)

    # --- 5. Save Blocks and Split into Train/Val ---
    os.makedirs(args.output_path, exist_ok=True)
    train_path = os.path.join(args.output_path, 'train')
    val_path = os.path.join(args.output_path, 'val')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    block_count = 0
    pbar = tqdm(total=len(grid_x) * len(grid_y), desc="Processing spatial blocks")
    for x0 in grid_x:
        for y0 in grid_y:
            pbar.update(1)
            x_min, x_max = x0, x0 + args.block_size
            y_min, y_max = y0, y0 + args.block_size

            point_indices = np.where(
                (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
                (points[:, 1] >= y_min) & (points[:, 1] < y_max)
            )[0]

            if len(point_indices) < args.min_points_per_block:
                continue

            np.random.shuffle(point_indices)
            for i in range(0, len(point_indices), args.num_points):
                chunk_indices = point_indices[i:i + args.num_points]

                if len(chunk_indices) < args.min_points_per_block:
                    continue

                block_data = all_data[chunk_indices, :]
                block_labels = labels[chunk_indices]
                block_to_save = np.hstack((block_data, block_labels[:, np.newaxis]))

                if np.random.rand() < args.val_split:
                    save_path = os.path.join(val_path, f'block_{block_count}.npy')
                else:
                    save_path = os.path.join(train_path, f'block_{block_count}.npy')
                
                np.save(save_path, block_to_save)
                block_count += 1
    pbar.close()
        
    print(f"Finished. Saved {block_count} total training/validation files.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare LAS data for DGCNN training with overlap and chunking.')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the input .las file.')
    parser.add_argument('--output_path', type=str, default='processed_data', help='Path to save processed data.')
    parser.add_argument('--num_features', type=int, default=7, choices=[4, 7], help='Number of features to use (4 for XYZ+I, 7 for all).')
    parser.add_argument('--num_classes', type=int, default=6, choices=[6, 7], help='Number of classes to prepare (6 main, or 7 to include other).')
    parser.add_argument('--block_size', type=float, default=10.0, help='Size of the square blocks to divide the point cloud into (in meters).')
    parser.add_argument('--num_points', type=int, default=4096, help='Maximum number of points per saved block file.')
    parser.add_argument('--overlap', type=float, default=0.5, help='Overlap percentage between blocks (0.0 to 1.0). Default is 0.5 for 50% overlap.')
    parser.add_argument('--min_points_per_block', type=int, default=100, help='Minimum number of points a chunk must have to be saved.')
    parser.add_argument('--val_split', type=float, default=0.2, help='Fraction of blocks to use for validation.')
    
    args = parser.parse_args()
    main(args)
