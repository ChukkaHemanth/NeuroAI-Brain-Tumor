# Real-Time Brain Tumor Diagnosis Using Deep Learning 🧠

## Overview
Early detection of brain tumors using MRI scans is crucial for improving treatment outcomes. This project provides an automated, lightweight, and highly accurate brain tumor segmentation system. Leveraging the **SegResNet** deep learning architecture and the **MONAI** framework, the model efficiently processes multi-modal MRI scans (T1, T1CE, T2, and FLAIR) to detect, classify, and segment tumor regions. 

An interactive **NeuroAI Diagnostic** dashboard built with Streamlit allows for real-time inference and visualization of the diagnostic results.

## Key Features
* **Automated Segmentation:** Accurately segments Whole Tumor (WT) regions from raw MRI data.
* **Lightweight & Efficient:** Optimized deep learning pipeline designed to reduce computational cost while enabling faster real-time analysis.
* **High Accuracy:** Achieved an overall segmentation accuracy (Mean Dice Score) of **89.6%** on testing data.
* **Interactive UI:** A highly intuitive Streamlit web application (`app.py`) for uploading MRI scans, running real-time diagnostics, and generating downloadable clinical reports.
* **Medical-Grade Preprocessing:** Automated standardization of diverse medical hardware outputs into uniform tensors using MONAI transforms (spacing, orientation, intensity normalization).

## Tech Stack
* **Languages:** Python
* **Deep Learning Frameworks:** PyTorch, MONAI (Medical Open Network for AI)
* **Web Framework:** Streamlit
* **Data Processing & Visualization:** NumPy, Pandas, Matplotlib
* **Model Architecture:** SegResNet (Encoder-Decoder framework with residual learning)

## Dataset
This model is trained from scratch on the **BraTS 2021-2022 (Brain Tumor Segmentation) Dataset**. The dataset consists of multi-parametric MRI scans, providing complex and diverse spatial contextual features essential for precise tumor localization.

## Project Structure
```text
├── app.py                      # Main Streamlit web application for the NeuroAI UI
├── colab.ipynb                 # Jupyter Notebook for data preprocessing and model training
├── Final-Result(3).ipynb       # Jupyter Notebook containing final model evaluation & testing metrics
├── segresnet_best.pth          # Saved PyTorch model weights containing the trained parameters
├── Real-Time brain tumor project.pptx  # Project presentation detailing architecture and outcomes
└── README.md                   # Project documentation
```

## Installation & Setup

**1. Clone the repository and navigate to the project directory:**
```bash
git clone <your-repo-link>
cd <your-repo-directory>
```

**2. Install the required dependencies:**
Ensure you have Python 3.8+ installed. Run the following command to install the necessary libraries:
```bash
pip install torch torchvision torchaudio monai streamlit numpy pandas matplotlib
```

**3. Run the Streamlit Application:**
Ensure the trained model weights (`segresnet_best.pth`) are in the correct directory as referenced in your code, then launch the app:
```bash
streamlit run app.py
```

**4. Usage:**
* Open the provided local URL in your web browser.
* Upload your multi-modal MRI scans (NIfTI format).
* View the real-time segmentation overlays, tumor volume calculations, and download the diagnostic text report.

## Outcomes
* Developed an automated brain tumor segmentation model using deep learning.
* Successfully trained for 60 epochs to learn complex tumor segmentation patterns.
* Achieved an 89.6% overall accuracy on unseen testing sets, providing a reliable proof-of-concept for clinical assistance.

## Team
**Group 18 (CSE C)**
* D Parimitha (1608-22-733-191)
* CH Hemanth (1608-22-733-165)
* T Meghana (1608-22-733-178)