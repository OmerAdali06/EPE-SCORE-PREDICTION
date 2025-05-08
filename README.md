# 🎯 EPE Score Prediction with ElasticNet

This project uses an ElasticNet regression model to predict English Proficiency Exam (EPE) scores based on 
academic-demographic features and field-collected data.

## 📁 Dataset Description

The dataset includes the following features:

- `1_note_mean`: Average academic score for the first semester
- `2_note_mean`: Average academic score for the second semester
- `Sex`: Categorical variable indicating gender (Female or Male)
- `EPE`: Final English Proficiency Exam score (target variable)

Missing values in `2_note_mean` were imputed using the sex-wise group mean. Outliers were identified using **Local Outlier Factor (LOF)** and mitigated via **Winsorization**.

## 🧠 Model Details

- **Algorithm**: ElasticNetCV (10-fold cross-validation)
- **Outlier Detection**: Local Outlier Factor (LOF)
- **Preprocessing**: One-hot encoding for categorical variables
- **Metrics**: Root Mean Squared Error (RMSE) and Coefficient of Determination (R²)

## 📊 Model Performance

| Model Variant | RMSE | R² Score |
|---------------|------|----------|
| Base Model    | 6.57 | -0.49    |
| Tuned Model   | 6.31 | -0.38    |

> ⚠️ Note: The low R² scores may be because of the lack of observation units in the dataset. Further data collection is recommended.

## 🚀 How to Run

1. Clone the repository.
2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the model:
    ```bash
    python model.py
    ```

## 🔮 Sample Prediction

```python
# Using the trained model
def EPEScorePrediction(1_note_mean):
    return 33.89 + 0.556 * 1_note_mean  #Simplified linear formula based on coefficients and intercept.
