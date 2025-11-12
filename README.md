# 🧠 MNIST Handwritten Digit Classification

This project demonstrates **supervised learning** for handwritten digit recognition using the **MNIST dataset**.  
It walks through the complete machine learning pipeline — from **data loading and preprocessing** to **model training, evaluation, and prediction**.

-----

## 📋 Project Overview

The **MNIST dataset** is a classic benchmark in machine learning consisting of **70,000 grayscale images** of handwritten digits (0–9), each of size **28×28 pixels**.  
The goal of this project is to build a model that accurately classifies digits from these images.

-----

## 🚀 Features

- Loads and preprocesses the MNIST dataset  
- Visualizes sample images and data distributions  
- Implements machine learning / deep learning models for classification  
- Evaluates model performance using accuracy and confusion matrix  
- Optionally saves and loads trained models  

---

## 🧩 Techniques Used

- Supervised Learning  
- Image Classification  
- Neural Networks (MLP/CNN)  
- Data Normalization and Flattening  
- Model Evaluation Metrics  

---

## 🛠️ Technologies Used

| Tool / Library | Purpose |
|-----------------|----------|
| **Python** | Programming Language |
| **NumPy** | Numerical Computations |
| **Pandas** | Data Handling |
| **Matplotlib / Seaborn** | Data Visualization |
| **TensorFlow / Keras** | Neural Network Model |
| **Scikit-learn** | Preprocessing & Evaluation |

---

## 📈 Model Workflow

1. Import Dependencies  
2. Load Dataset (MNIST)  
3. Preprocess Data (Normalization, Reshape)  
4. Build Model (Sequential Neural Network)  
5. Compile & Train Model  
6. Evaluate Accuracy  
7. Visualize Results  

---

## 🎯 Results

- Achieved **~98% test accuracy** (depending on parameters and architecture).  
- Successfully classifies digits **0–9** with high precision.

---

## 📁 File Structure

```
mnist-classification/
│
├── mnist-classification.ipynb   # Main Jupyter Notebook
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies (optional)
```

---

## ⚙️ How to Run

1. **Clone the repository** or download the project:
   ```bash
   git clone https://github.com/Shivansh3270/mnist-classification.git
   cd mnist-classification
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy pandas matplotlib seaborn tensorflow scikit-learn
   ```

3. **Open the Jupyter Notebook:**
   ```bash
   jupyter notebook mnist-classification.ipynb
   ```

4. **Run all cells** to train and evaluate the model.

---

## 🧪 Example Output

```
Training Accuracy: 99.2%
Test Accuracy: 98.1%
```

---

## 📚 Future Improvements

- Add **Convolutional Neural Networks (CNNs)** for higher accuracy  
- Use **Keras Tuner** for hyperparameter optimization  
- Deploy model using **Flask** or **Streamlit**

---

## 👥 Contributors

| Name | Responsibilities  | GitHub |
|------|-------|--------|
| **Shivansh Chaurasia** | Model selection and training | [@Shivansh Chaurasia](https://github.com/Shivansh3270) |
| **Nikita Tripathi** | Model evaluation, prediction |[@Nikita Tripathi](https://github.com/nikitatri) |
| **Avinash Saini** | Data collection, preprocessing | [@Avinash Saini](https://github.com/Avinash-Saini090) |

