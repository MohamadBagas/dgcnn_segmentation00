import os
import argparse
import laspy
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from model import DGCNNSemSeg

def main(args):
    # --- Setup ---
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # --- Load Model ---
    print("Loading trained model...")
    model = DGCNNSemSeg(num_classes=args.num_classes, num_features=args.num_features, k=args.k).to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Load and Preprocess Data ---
    print(f"Loading and preprocessing point cloud for inference with {args.num_features} features...")
    try:
        source_las = laspy.read(args.file_path)
        
        has_classification = "classification" in source_las.point_format.dimension_names
        if has_classification:
            print("Found 'classification' attribute. Accuracy will be calculated.")
            original_labels = np.array(source_las.classification)
        else:
            print("No 'classification' attribute found. Skipping accuracy calculation.")
            original_labels = None

    except Exception as e:
        print(f"Error loading LAS file: {e}")
        return

    points = np.vstack((source_las.x, source_las.y, source_las.z)).transpose()
    try:
        if args.num_features == 4:
            features = source_las.intensity.reshape(-1, 1)
        else:
            features = np.vstack((
                source_las.intensity,
                source_las.Range,
                source_las.Ring,
                source_las.AngleOfIncidence
            )).transpose()
    except AttributeError as e:
        print(f"Feature extraction failed: {e}. Ensure LAS file has required attributes.")
        return

    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    all_data = np.hstack((points, features_normalized))
    
    # --- Divide into Blocks for Inference (Sliding Window) ---
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    stride = args.block_size * (1 - args.overlap)

    grid_x = np.arange(min_coords[0], max_coords[0], stride)
    grid_y = np.arange(min_coords[1], max_coords[1], stride)

    # --- Run Inference with Averaging ---
    logit_sum = np.zeros((len(points), args.num_classes), dtype=np.float32)
    point_counts = np.zeros(len(points), dtype=np.int32)

    pbar = tqdm(total=len(grid_x) * len(grid_y), desc="Running inference on blocks")
    with torch.no_grad():
        for x0 in grid_x:
            for y0 in grid_y:
                pbar.update(1)
                x_min, x_max = x0, x0 + args.block_size
                y_min, y_max = y0, y0 + args.block_size
                
                in_block_indices = np.where(
                    (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
                    (points[:, 1] >= y_min) & (points[:, 1] < y_max)
                )[0]

                if len(in_block_indices) < 10: continue

                np.random.shuffle(in_block_indices)
                for i in range(0, len(in_block_indices), args.num_points):
                    chunk_indices = in_block_indices[i:i + args.num_points]
                    
                    if len(chunk_indices) < 10: continue
                        
                    block_data_sampled = all_data[chunk_indices, :]
                    
                    center = np.mean(block_data_sampled[:, :3], axis=0)
                    block_data_sampled[:, :3] -= center
                    points_tensor = torch.from_numpy(block_data_sampled).float().unsqueeze(0).to(device)

                    pred_logits = model(points_tensor)
                    
                    logit_sum[chunk_indices] += pred_logits.squeeze(0).cpu().numpy()
                    point_counts[chunk_indices] += 1
    pbar.close()
    print("Inference complete. Averaging predictions...")

    final_preds_0_indexed = np.zeros(len(points), dtype=np.int32)
    valid_points_mask = point_counts > 0
    final_preds_0_indexed[valid_points_mask] = np.argmax(logit_sum[valid_points_mask] / point_counts[valid_points_mask, np.newaxis], axis=1)

    # --- Calculate and Print Accuracy (if original labels exist) ---
    if original_labels is not None:
        original_labels_0_indexed = original_labels - 1
        if args.num_classes == 7:
            original_labels_0_indexed[original_labels > 6] = 6
            original_labels_0_indexed[original_labels == 0] = 6
        
        correct_predictions = np.sum(final_preds_0_indexed == original_labels_0_indexed)
        accuracy = correct_predictions / len(original_labels)
        print(f"\nOverall Accuracy on the full point cloud: {accuracy * 100:.2f}%\n")

        # --- Plot and Save Confusion Matrix ---
        class_names_map = {
            0: "Asphalt Road", 1: "Brick Road", 2: "Building",
            3: "Grass", 4: "Poles", 5: "Trees", 6: "Other"
        }
        class_labels = [class_names_map.get(i, f"Class {i}") for i in range(args.num_classes)]
        
        cm = confusion_matrix(original_labels_0_indexed, final_preds_0_indexed, labels=range(args.num_classes))
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        plot_path = os.path.join(os.path.dirname(args.output_path), 'confusion_matrix_report.png')
        plt.savefig(plot_path)
        print(f"Confusion matrix plot saved to {plot_path}")

    # --- Save Results to new LAS/LAZ file ---
    print(f"Saving segmented point cloud to {args.output_path}...")
    
    new_las = laspy.LasData(source_las.header)
    new_las.points = source_las.points.copy()
    
    new_las.add_extra_dim(laspy.ExtraBytesParams(name="prediction", type=np.uint8, description="DGCNN classification prediction"))
    
    new_las.prediction = final_preds_0_indexed.astype(np.uint8) + 1
    
    new_las.write(args.output_path)
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference with a trained DGCNN model using overlap averaging.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model (.pth) file.')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the input .las file for segmentation.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the segmented .las or .laz file.')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of classes the model was trained on.')
    parser.add_argument('--num_features', type=int, default=7, choices=[4, 7], help='Number of features to use (must match training).')
    parser.add_argument('--num_points', type=int, default=4096, help='Number of points to process at a time.')
    parser.add_argument('--block_size', type=float, default=10.0, help='Block size for processing.')
    parser.add_argument('--overlap', type=float, default=0.5, help='Overlap used during inference. Should match training. Default is 0.5.')
    parser.add_argument('--k', type=int, default=20, help='k for k-NN in DGCNN.')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for training.')
    
    args = parser.parse_args()
    main(args)
