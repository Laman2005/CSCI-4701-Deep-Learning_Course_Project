# Cardiomegaly Detection from Chest X-Ray Images using Deep Learning

**Course:** CSCI-4701 Deep Learning Course_Project
**Semester:** Spring 2026

---

# Project Goal

The goal of this project is to develop a deep learning system that can automatically detect **cardiomegaly (enlarged heart)** from chest X-ray images. Cardiomegaly is an important clinical indicator associated with several cardiovascular diseases. Detecting it accurately from radiographic images can help support medical diagnosis and assist clinicians in identifying potential heart conditions.

The project aims to build a **complete machine learning pipeline using PyTorch** that loads medical imaging data, preprocesses the images, trains a convolutional neural network, and evaluates the model’s predictive performance. In addition to measuring standard performance metrics, the project also analyzes whether the model behaves differently for **male and female patients**, which helps identify potential biases in the dataset or the trained model.

---

# Problem Description

Chest X-ray interpretation is a challenging task that requires expert knowledge from radiologists. With the increasing availability of medical imaging datasets, deep learning models have shown strong potential for assisting with automated image analysis.

However, building reliable medical AI systems involves several challenges:

* Medical datasets often contain **label uncertainty**
* Class distributions may be **imbalanced**
* Models may learn **spurious correlations**
* Predictions must remain **interpretable and trustworthy**

This project focuses on addressing these challenges by building a robust training pipeline and analyzing both **model performance and interpretability**.

---

# Dataset

This project uses the **CheXpert-v1.0-small dataset**, which is a subset of the CheXpert dataset released by Stanford. The dataset used in this project is accessed through Kaggle and is available at: https://www.kaggle.com/datasets/ashery/chexpert 

The dataset contains chest X-ray images together with labels extracted automatically from radiology reports. Each record includes:

* X-ray image
* Patient age
* Patient sex
* Image view
* Diagnostic labels for multiple thoracic conditions

For this project, the task is simplified to **binary classification**, where the model predicts whether **cardiomegaly is present or not**.

Some labels in the dataset are marked as **uncertain (-1)**. In this project these uncertain labels are treated as negative cases to simplify training.

---

# Approach

The system is implemented in **PyTorch** and follows a modular machine learning pipeline.

### 1. Data Loading

A custom PyTorch dataset class loads X-ray images, labels, and metadata from the dataset CSV files.

### 2. Data Preprocessing

Images are resized and normalized before being passed to the neural network. Data augmentation techniques such as random cropping, flipping, and rotation are applied during training to improve model generalization.

### 3. Model Architecture

A pretrained convolutional neural network (**EfficientNet-B4**) is used through transfer learning. The final classification layer is replaced with a single output neuron for binary classification.

### 4. Training

The model is trained using binary cross-entropy loss with class weighting to handle class imbalance. The **AdamW optimizer** is used for training, and learning rate scheduling is applied to stabilize optimization.

### 5. Evaluation

Model performance is evaluated on a validation dataset using several metrics including **accuracy, ROC-AUC, and F1-score**.

HERE

---

# Model Interpretability

In medical applications it is important to understand **why the model makes certain predictions**. For this reason, interpretability methods were applied to visualize which parts of the image influence the model’s decision.

The project uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to generate heatmaps that highlight regions of the X-ray image that contribute most to the prediction.

These heatmaps are overlaid on the original images to produce **focus maps**, which allow us to verify whether the model is focusing on clinically relevant areas such as the **cardiac region**.

This helps ensure that the model is learning meaningful patterns rather than relying on unrelated image artifacts.

---


# Experimental Results

The trained model achieved the following results on the validation dataset:

| Group   | Accuracy | ROC-AUC | F1 Score |
| ------- | -------- | ------- | -------- |
| Overall | 0.782    | 0.690   | 0.549    |
| Male    | 0.789    | 0.667   | 0.557    |
| Female  | 0.774    | 0.706   | 0.538    |

These results show that the model is able to detect cardiomegaly reasonably well, although there is still room for improvement.

---

# Results Analysis

Several observations can be made from the experimental results.

First, the model achieves moderate predictive performance. The ROC-AUC score indicates that the model is able to distinguish positive and negative cardiomegaly cases better than random guessing, but the classification task remains challenging.

Second, the dataset shows some class imbalance, which may affect the F1-score and prediction stability. Handling uncertainty labels and limited training samples may also influence performance.

Third, the fairness analysis shows only small differences between male and female patients. While performance is relatively similar across both groups, further experiments with larger samples would be needed to confirm whether any bias exists.

Finally, Grad-CAM visualizations indicate that the model often focuses on the **region surrounding the heart**, which suggests that the network is learning relevant visual features rather than unrelated artifacts.

---

# Milestone 2 Improvements

For Milestone 2, the model will be further improved through several steps:

* Fine-tuning additional layers of the pretrained network
* Improving **accuracy, ROC-AUC, and F1-score**
* Performing **hyperparameter tuning**
* Conducting deeper **error analysis**
* Expanding fairness analysis across more patient attributes

These improvements aim to produce a more reliable and better-performing model.

---

# Repository Structure

```
.
├── notebook/
│   └── cardiomegaly_pipeline.ipynb
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluation.py
├── results/
└── README.md
```

The repository is organized so that the core functionality is implemented as **modular Python code**, while the main notebook imports these modules to run the full experiment.

---

# How to Install Dependencies

The project can be run using **Google Colab**.

Required Python libraries include:

* PyTorch
* torchvision
* pandas
* numpy
* matplotlib
* scikit-learn
* kagglehub

Dependencies can be installed using pip:

```
pip install torch torchvision pandas numpy matplotlib scikit-learn kagglehub
```

---

# Running the Project

1. Open the main notebook in **Google Colab**
2. Run all cells from top to bottom
3. The notebook will automatically:

   * Download the dataset
   * Train the model
   * Evaluate performance
   * Generate visualizations and results

Running the notebook end-to-end reproduces all reported results.

---

# Team Member Contributions

| Team Member     | Contribution                                                                                             |
| --------------- | -------------------------------------------------------------------------------------------------------- |
| Laman Panakhova | Data preprocessing, dataset implementation, model training pipeline, visualizations, experiment analysis |

---

# Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* KaggleHub

---

# Reproducibility

The repository includes a single **main Jupyter notebook** that runs end-to-end on Google Colab. The notebook imports modular Python code from the repository and reproduces all reported experimental results.
