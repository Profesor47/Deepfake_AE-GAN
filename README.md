# Autoencoder (AE) + GAN on CelebA 🎭

**Deep Learning | M.Tech Mini-Project | Context: Deepfake Case Study**

This repository contains the implementation of a Deep Learning mini-project developed for an M.Tech curriculum. The project focuses on a **Deepfake Case Study** by training and evaluating an **Autoencoder (AE)** followed by a **Generative Adversarial Network (GAN)** using the CelebA dataset.

## 📖 Overview
The primary objective of this project is to explore deep generative models and their implications in deepfake technology. By building an Autoencoder and a GAN from scratch using PyTorch, the project investigates facial image reconstruction, dimensionality reduction, and synthetic face generation. 

### Key Features:
- **Autoencoder (AE)**: Learns compressed latent representations of faces and reconstructs them.
- **Generative Adversarial Network (GAN)**: Generates highly realistic artificial face images.
- **Latent Space Analysis**: Uses **PCA** and **t-SNE** to visualize the learned embeddings.
- **Quality Metrics**: Computes **MSE (Mean Squared Error)** and **SSIM (Structural Similarity Index)** to objectively evaluate reconstruction quality.

## 🧰 Tech Stack & Libraries
- **Language**: Python 3.12
- **Framework**: PyTorch (`torch`, `torch.nn`, `torchvision`)
- **Data Engineering & Metrics**: NumPy, Scikit-Learn (`PCA`, `t-SNE`, `mean_squared_error`), Scikit-Image (`ssim`)
- **Environment**: Optimized for Kaggle Notebooks / Jupyter Lab with CUDA (GPU) support.

## 📊 Dataset
- **CelebFaces Attributes (CelebA) Dataset**: A large-scale face attributes dataset with over 200,000 celebrity images.
- **Kaggle Path**: `/kaggle/input/datasets/jessicali9530/celeba-dataset`
- **Preprocessing**: Images are resized to `64x64`, center-cropped, and normalized to the `[-1, 1]` range.

## ⚙️ Hyperparameters
| Parameter | Value |
| :--- | :--- |
| **Image Size** | 64x64 (3 Channels) |
| **Latent Dimension (AE)** | 128 |
| **Noise Dimension (GAN - NZ)** | 100 |
| **Batch Size** | 128 |
| **AE Epochs**| 20 |
| **GAN Epochs** | 30 |
| **Learning Rate (AE)** | 1e-3 |
| **Learning Rate (GAN - G & D)** | 2e-4 |
| **Optimizer** | Adam (Beta1: 0.5, Beta2: 0.999) |

## 🚀 How to Run

1. **Kaggle Environment (Recommended):**
   - Import the `deepreview3-1 (1).ipynb` notebook into Kaggle.
   - Attach the **CelebA** dataset (`jessicali9530/celeba-dataset`) to the notebook.
   - Ensure the **GPU (CUDA)** accelerator is turned on.
   - Run the cells sequentially.

2. **Local Environment:**
   ```bash
   # Clone the repository
   git clone <your-repo-url>
   cd <your-repo-folder>
   
   # Install dependencies
   pip install torch torchvision scikit-learn scikit-image matplotlib numpy
   
   # Launch Jupyter
   jupyter notebook "deepreview3-1 (1).ipynb"
   ```
   *Note: Make sure to update the `DATA_DIR` path in the hyperparameter configuration cell (Cell 2) to point to your local CelebA dataset directory.*

## 📈 Evaluation & Results
The notebook natively outputs graphical comparisons of:
1. Real vs. Reconstructed Images (Autoencoder).
2. Progression of GAN-generated faces across epochs.
3. PCA and t-SNE scatter plots illustrating the structure of the 128-dimensional latent space.

---
*Disclaimer: This project is strictly for educational and research purposes as part of an M.Tech curriculum on Deep Learning and AI ethics concerning deepfakes.*
