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

#### Evaluation Results 

![Evaluation Result 1](https://raw.githubusercontent.com/Laman2005/CSCI-4701-Deep-Learning_Course_Project/main/Screenshot%202026-03-14%20113918.png)

![Evaluation Result 2](https://raw.githubusercontent.com/Laman2005/CSCI-4701-Deep-Learning_Course_Project/main/Screenshot%202026-03-14%20114409.png)

![Evaluation Result 3](https://raw.githubusercontent.com/Laman2005/CSCI-4701-Deep-Learning_Course_Project/main/Screenshot%202026-03-14%20114454.png) 
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

Several observations can be made from the results of the cardiomegaly detection model during training and evaluation.

First, the model shows moderate performance on the validation dataset. The ROC-AUC score indicates that the model can distinguish between cardiomegaly and non-cardiomegaly cases better than random guessing. This means the model has learned some useful patterns from the chest X-ray images. However, the performance is still not very high, which shows that detecting cardiomegaly from X-rays is a difficult task. The differences between a normal heart and an enlarged heart can sometimes be very small, and factors like patient position, image quality, and anatomical differences can also affect the images.

Another important factor is the class imbalance in the dataset. There are fewer cardiomegaly cases compared to normal cases, which may cause the model to predict the majority class more often. Although class weighting was used to reduce this issue, the imbalance may still affect the F1-score and lead to some incorrect predictions. In addition, the CheXpert dataset includes uncertain labels that were treated as negative in this project. While this makes the training process easier, it may also introduce some noise into the labels and reduce the overall accuracy of the model.

The fairness analysis across patient sex shows that the model performs similarly for male and female patients. The differences in accuracy and ROC-AUC scores are small, which suggests that the model does not strongly favor one group over the other. However, these results should still be interpreted carefully because differences in sample size or other factors like age and imaging view might influence the results. More analysis with larger datasets would help better evaluate fairness.

The Grad-CAM visualization also helps us understand how the model makes its predictions. In many cases, the highlighted areas are close to the cardiac silhouette, which is the region that radiologists usually focus on when checking for cardiomegaly. This suggests that the model is learning meaningful features from the images. However, in some cases the highlighted regions extend outside the heart area, which may indicate that the model is sometimes influenced by other parts of the image.

Some methods used in this project worked well. Transfer learning with a pretrained convolutional neural network helped the model learn useful features more quickly than training from scratch. Data augmentation techniques such as cropping, flipping, and rotation also helped improve the model’s ability to generalize by creating variations of the training images.

However, there are also some limitations. The dataset used in the experiment was smaller due to computational limitations, which may limit the model’s ability to learn stronger patterns. In addition, the task was simplified to a binary classification problem, even though chest X-rays can contain multiple medical conditions at the same time.

Overall, the results show that the model is able to learn useful information from chest X-ray images and make reasonable predictions about cardiomegaly. At the same time, challenges such as class imbalance, uncertain labels, and limited data still affect the model’s performance. Future improvements such as better tuning and using more data could help improve the model further.

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
