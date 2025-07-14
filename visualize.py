import argparse
import open3d as o3d
import laspy
import numpy as np

def main(args):
    """
    Loads and visualizes a LAS/LAZ file by its classification attribute.
    """
    print(f"Attempting to load and visualize {args.file_path}...")
    
    try:
        las = laspy.read(args.file_path)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        
        # Check if the file has the 'prediction' attribute. If not, use 'classification'.
        if 'prediction' in las.point_format.dimension_names:
            print("Coloring by 'prediction' attribute.")
            labels = las.prediction
        elif 'classification' in las.point_format.dimension_names:
            print("Coloring by 'classification' attribute.")
            labels = las.classification
        else:
            print("No classification or prediction attribute found. Displaying without color.")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            o3d.visualization.draw_geometries([pcd])
            return

        # Define a color map for visualization
        color_map = {
            1: [178, 34, 34],   # Asphalt (Firebrick)
            2: [210, 105, 30],  # Brick (Chocolate)
            3: [70, 130, 180],  # Building (SteelBlue)
            4: [34, 139, 34],   # Grass (ForestGreen)
            5: [255, 255, 0],   # Poles (Yellow)
            6: [139, 69, 19],   # Trees (SaddleBrown)
            7: [128, 128, 128]  # Other/Unclassified (Gray)
        }
        default_color = [255, 255, 255] # White

        colors = np.array([color_map.get(label, default_color) for label in labels])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

        if not pcd.has_points():
            print("Error: The point cloud is empty or could not be read.")
            return

    except Exception as e:
        print(f"Failed to read or process point cloud file: {e}")
        return
        
    print("Displaying point cloud. Press 'Q' in the window to close.")
    o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize a point cloud file by classification.')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the point cloud file (.las, .laz).')
    
    args = parser.parse_args()
    main(args)
