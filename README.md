
# 🚢 Titanic Survival Prediction - Data Mining Project

Welcome to the **Titanic Survival Prediction** project, developed as part of a data mining course. This project focuses on predicting survival outcomes from the famous Titanic dataset using various data preprocessing techniques and a deep learning model.

---

## 📌 Project Workflow

### 🎯 1. Feature Selection
We applied a **Random Forest Classifier** to identify the most influential features in the dataset. This helps reduce dimensionality and enhances model performance.

### 🔡 2. Categorical Encoding
Categorical columns such as `Sex`, `Cabin`, and `Ticket` were transformed into numerical format using **Label Encoding** to ensure compatibility with machine learning algorithms.

### 📏 3. Data Normalization
Features were scaled using **StandardScaler** to bring them to a comparable range, ensuring optimal convergence during model training.

### 🧠 4. Model Training
We designed a **Neural Network** using **TensorFlow/Keras** with:
- Two hidden layers
- ReLU activation functions
- A final sigmoid layer for binary classification (Survived / Not Survived)

### 📈 5. Evaluation
- Model accuracy was calculated on the test dataset.
- A **Confusion Matrix** was generated for performance insights.
- Network architecture was visualized using plotting tools.

### ⚖️ 6. Fairness Check (Chi-Square Test)
To ensure statistical similarity between training and testing distributions, we performed a **Chi-Square Test** on selected features. This helps validate the fairness of our train-test split.

---

## 📁 File Structure

```bash
├── random_forest_feature_selection.py   # Feature selection using Random Forest
├── neural_network_model.py              # Model training and evaluation
├── nn_structure_plot.py                 # Visualizing the network structure
```

---

## 📊 Results
The trained model achieved solid accuracy on the test data, with a well-balanced performance across different passenger groups. The visualized network offers clear insight into its architecture.

---

## 🚀 Tools & Libraries

- Python 🐍
- Pandas & NumPy 📊
- TensorFlow / Keras 🤖
- Matplotlib & Seaborn 📉
- Scikit-learn ⚙️

---

## 🤝 Acknowledgments

Special thanks to the data mining course instructors and the vibrant open-source community behind the Titanic dataset and the amazing libraries used in this project.

---

Feel free to explore the code, run experiments, or build upon this work!

> _"Survival is not about luck, it's about preparation."_ 🧭
