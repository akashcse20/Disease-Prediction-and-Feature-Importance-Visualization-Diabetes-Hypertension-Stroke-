---

# 🌟 **Disease Prediction & Feature Importance Visualization** 🌟  
### **Predicting Diabetes, Hypertension & Stroke Using Health Data**  
This repository offers a robust solution for predicting **Diabetes**, **Hypertension**, and **Stroke** using health-related data, while also visualizing the importance of features contributing to these predictions. The project uses **LightGBM (LGBM)** for model training and evaluation, delivering accurate predictions with advanced machine learning techniques.

---

## 🚀 **Overview**
This project leverages health data to predict the likelihood of three common diseases:
- **Diabetes**
- **Hypertension**
- **Stroke**

By applying state-of-the-art **LightGBM** (LGBM) regression models, the dataset undergoes preprocessing, including label encoding, handling missing values, and data normalization. The model is trained using **K-Fold Cross-Validation**, ensuring its robustness. 

After training, **feature importance** is visualized to show which variables contribute most to each disease's prediction, helping us understand the key factors driving these health conditions.

---

## 📦 **Installation**

To set up the environment and install all necessary dependencies, follow these simple steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/akashcse20/Disease-Prediction-and-Feature-Importance-Visualization-Diabetes-Hypertension-Stroke-.git
   cd Disease-Prediction-and-Feature-Importance-Visualization-Diabetes-Hypertension-Stroke-
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📋 **Dependencies**
To run this project, make sure the following Python packages are installed:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `lightgbm`
- `scikit-learn`
- `imblearn`
- `tensorflow`
- `tqdm`
- `category_encoders`

To install all dependencies at once, simply run:
```bash
pip install -r requirements.txt
```

---

## 🔥 **Data Preparation**
The dataset contains health-related features and labels for **Diabetes**, **Hypertension**, and **Stroke**. Key preprocessing steps include:
1. **Label Encoding**: Converts categorical features to numeric values.
2. **Missing Value Handling**: Fills missing values with a placeholder (`'N'`).
3. **Data Splitting**: The dataset is split into **training** (75%) and **testing** (25%) sets.

---

## 📊 **Model Training**
The project employs **LightGBM (LGBM)**, a powerful gradient boosting framework. The model is trained using **K-Fold Cross-Validation** to evaluate its performance in various splits of the dataset. Hyperparameters such as **learning rate**, **max depth**, and **regularization** are optimized.

### Key Model Training Code:
```python
from sklearn.model_selection import KFold
import lightgbm as lgbm

def fit_lgbm(X, y, cv, params=None, verbose=50):
    models = []
    oof_pred = np.zeros_like(y, dtype=np.float)
    for i, (idx_train, idx_valid) in enumerate(cv): 
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]
        clf = lgbm.LGBMRegressor(**params)
        clf.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], early_stopping_rounds=100, verbose=verbose)
        pred_i = clf.predict(x_valid)
        oof_pred[idx_valid] = pred_i
        models.append(clf)
    return oof_pred, models
```

---

## 💡 **Feature Importance Visualization**
Visualizing the **importance of features** helps identify which health-related factors contribute most to disease predictions. We use a **boxen plot** to display the importance of each feature, sorted by their total contribution.

### Feature Importance Visualization Code:
![Image](https://github.com/user-attachments/assets/d9478eed-4876-4f57-9567-a37edd064ab0)

![Image](https://github.com/user-attachments/assets/96cb06fd-f029-4703-8eb7-9d7f01cae72b)

![Image](https://github.com/user-attachments/assets/b2751a3a-8682-4e07-8af6-98af2726114e)

---

## 🏆 **Evaluation**

The model's performance is assessed through **classification metrics**, including **precision**, **recall**, **F1-score**, and **accuracy**. We compare the predicted values with the true labels to evaluate its predictive power.

### Example of Evaluation Output:

![Image](https://github.com/user-attachments/assets/03bc895d-5383-482d-9691-708558450384)

![Image](https://github.com/user-attachments/assets/52b3778e-86d2-4364-acf8-8f57bd3f4200)

![Image](https://github.com/user-attachments/assets/5f81e774-1460-4c6c-9175-ee33c1f7f59e)


---

## 📈 **Results**

The results include:
- **Visualizations** of predicted values and **feature importance** for each disease.
- **Histograms** for predicted values (train vs. test).
- **Feature Importance** plots, showcasing the most influential features for each disease prediction.

### Example of Prediction Comparison:

**Diabetes**
```
              precision    recall  f1-score   support

          No     0.7462    0.7075    0.7263      8825
         Yes     0.7226    0.7599    0.7408      8848

    accuracy                         0.7338     17673
   macro avg     0.7344    0.7337    0.7336     17673
weighted avg     0.7344    0.7338    0.7336     17673

```
**Hypertension**
```
              precision    recall  f1-score   support

          No     0.6976    0.6001    0.6452      7715
         Yes     0.7205    0.7985    0.7575      9958

    accuracy                         0.7119     17673
   macro avg     0.7090    0.6993    0.7013     17673
weighted avg     0.7105    0.7119    0.7085     17673

```
**Stroke**
   ```
              precision    recall  f1-score   support

   macro avg     0.4687    0.5000    0.4839     17673
weighted avg     0.8789    0.9375    0.9072     17673


```

---

## 🎯 **Usage**

To use this project:
1. Clone the repository.
2. Prepare your own health dataset in the required format.
3. Run the **notebooks** or execute the scripts in the appropriate order to:
   - Train the model.
   - Make predictions.
   - Visualize feature importance.

### Running the Notebook:
Simply open the Jupyter notebooks in the repository and execute the cells to run the code step by step.

---

## 📝 **License**

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---
