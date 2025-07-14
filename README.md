# **DGCNN for Point Cloud Semantic Segmentation**

This project provides a complete, flexible pipeline to train a Dynamic Graph Convolutional Network (DGCNN) for semantic segmentation of LiDAR point clouds. This final version uses a stable, deep architecture and allows you to easily experiment with different feature sets and class configurations.

### **Project Structure**

Create this folder structure for your project:

dgcnn\_segmentation/  
│  
├── data/  
│   └── your\_campus\_data.las  \#\<-- Place your raw data file here  
│  
├── processed\_data/           \#\<-- This will be created by the prep script  
│  
├── checkpoints/              \#\<-- Trained models will be saved here  
│  
├── results/                  \#\<-- Segmented clouds and reports will be saved here  
│  
├── prepare\_data.py           \#\<-- Script to process the raw .las file  
├── dataset.py                \#\<-- PyTorch custom Dataset  
├── model.py                  \#\<-- The DGCNN model architecture  
├── train.py                  \#\<-- The main training script  
├── inference.py              \#\<-- Script to run segmentation on new data  
└── visualize.py              \#\<-- Utility to view the results

### **Step 1: Environment Setup**

Before running any scripts, you need to set up a Python environment with the required libraries.

1. **Create a virtual environment (recommended):**  
   python \-m venv venv  
   source venv/bin/activate  \# On Windows, use \`venv\\Scripts\\activate\`

2. **Install libraries:**  
   pip install torch torchvision torchaudio  
   pip install numpy laspy open3d scikit-learn tqdm matplotlib seaborn

   *Note: For GPU acceleration, ensure you install a version of PyTorch compatible with your CUDA toolkit.*

### **Step 2: Data Preparation**

This crucial step reads your raw .las file and prepares it for training. You can now choose the number of features and classes directly from the command line.

1. Place your data file your\_campus\_data.las inside the data/ directory.  
2. **Delete any old processed\_data folder** to ensure you're using a clean dataset.  
3. Run the prepare\_data.py script. Use the \--num\_features and \--num\_classes flags to configure your dataset.  
   **Example: Prepare data with 7 features and 6 classes:**  
   python prepare\_data.py \--file\_path data/your\_campus\_data.las \--num\_features 7 \--num\_classes 6

   **Example: Prepare data with 4 features and 7 classes (including 'other'):**  
   python prepare\_data.py \--file\_path data/your\_campus\_data.las \--num\_features 4 \--num\_classes 7

### **Step 3: Training the Model**

Now you can train the DGCNN model. A live plot will appear to show you the training progress, and the final graph will be saved as training\_progress.png.

**Crucially, the \--num\_features and \--num\_classes flags must match the ones you used in Step 2\.**

**Example: Train a model on 7 features and 6 classes:**

python train.py \--data\_path processed\_data/ \--num\_features 7 \--num\_classes 6 \--use\_gpu

**Example: Train a model on 4 features and 7 classes:**

python train.py \--data\_path processed\_data/ \--num\_features 4 \--num\_classes 7 \--use\_gpu

The model with the best validation performance will be saved in the checkpoints/ directory as best\_model.pth.

### **Step 4: Inference**

Once the model is trained, use it to segment a point cloud. This script calculates accuracy and saves predictions to a new prediction attribute.

**Again, make sure \--num\_features and \--num\_classes match the parameters you used for training.**

**Example: Run inference with a model trained on 7 features and 6 classes:**

python inference.py \--model\_path checkpoints/best\_model.pth \--file\_path data/your\_campus\_data.las \--output\_path results/segmented\_cloud.las \--num\_features 7 \--num\_classes 6 \--use\_gpu

**Example: Run inference with a model trained on 4 features and 7 classes:**

python inference.py \--model\_path checkpoints/best\_model.pth \--file\_path data/your\_campus\_data.las \--output\_path results/segmented\_cloud.las \--num\_features 4 \--num\_classes 7 \--use\_gpu

This will create segmented\_cloud.las and confusion\_matrix\_report.png in the results/ directory.

### **Step 5: Visualization**

To see your final result, you can use software like CloudCompare or the provided visualize.py script.

python visualize.py \--file\_path results/segmented\_cloud.las  
