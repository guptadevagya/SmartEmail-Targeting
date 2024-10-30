# Predicting User Engagement with TV Promotion Emails

This project applies machine learning techniques to predict which users are likely to click on a promotional email for a TV. By analyzing user demographics and browsing history, the model identifies potential customers, optimizing marketing efforts and enhancing customer experience. The analysis includes data integration, feature engineering, model development, and evaluation.

---

## 📑 Project Overview

In the competitive online retail space, targeted marketing is essential for maximizing engagement and minimizing unnecessary outreach. This project utilizes historical user data and interaction logs to model and predict user behavior concerning promotional emails. By identifying users more likely to engage with the TV promotion, the project aims to improve marketing efficiency and customer satisfaction.

---

## 🎯 Objectives

1. **Data Integration**: Merge user demographics with browsing history to form a comprehensive dataset.
2. **Feature Engineering**: Extract and create meaningful features from raw data, such as user activity patterns and page visit metrics.
3. **Model Development**: Implement a logistic regression model to predict user engagement with the promotional email.
4. **Evaluation**: Assess the model's performance using accuracy metrics and refine it through parameter tuning.
5. **Prediction**: Generate predictions on unseen data to identify potential customers.
6. **Insights**: Interpret model results to understand the key factors influencing user engagement.

---

## 📁 Project Structure

### Files
- **`main.py`**: Contains the `UserPredictor` class and model implementation.
- **`tester.py`**: Script to test the model with provided datasets.
- **`data/`**: Directory containing the datasets.
  - `train_users.csv`
  - `train_logs.csv`
  - `train_clicked.csv`
  - `test1_users.csv`
  - `test1_logs.csv`
  - `test1_clicked.csv`
- **`results.json`**: Output file with evaluation results.

---

## 🛠️ Feature Engineering Details

- **`total_minutes`**: Total time a user spent on the website.
- **`avg_session_duration`**: Average time per browsing session.
- **`days_since_last_visit`**: Number of days since the user's last website visit.
- **`visits_to_tv`**: Number of times the user visited the TV product page.
- **`time_on_tv`**: Total time spent on the TV product page.
- **`unique_pages`**: Number of unique pages the user has visited.
- **`visited_tv`**: Indicator (0 or 1) of whether the user has visited the TV page.
- **`part_of_day`**: The time of day when the user is most active on the website.

---

## 🔧 Model Implementation

### Data Preprocessing:
- **Handling missing values** and data type conversions.
- **Standardizing numerical features** using `StandardScaler`.
- **Generating polynomial features** for interaction terms with `PolynomialFeatures`.
- **Encoding categorical variables** using `OneHotEncoder`.

### Pipeline Setup:
- **Utilizing `Pipeline` and `ColumnTransformer`** from `scikit-learn` for streamlined preprocessing and model training.

### Logistic Regression Model:
- Configured with **L1 regularization (`penalty='l1'`)** to perform feature selection.
- Set **`class_weight='balanced'`** to handle class imbalance.
- **Solver set to 'liblinear'** for compatibility with L1 penalty.

---

## 🚀 Running the Code

**Dependencies**: Ensure the following libraries are installed: `pandas`, `numpy`, `scikit-learn`.

This command trains the model using the training data and evaluates it on the test dataset.

```bash
python3 tester.py
```

## 📊 Output Files

- **`results.json`**: Contains the model's accuracy and score.

```json
{
  "accuracy": 87.90666666666667,
  "date": "04/28/2024",
  "latency": 0.3333721160888672
}
```

## 📈 Results and Observations

The logistic regression model effectively predicts user engagement with the TV promotion email, achieving an accuracy exceeding 75% on the test dataset.

### Key Observations:

#### Influential Features:
- **Users with prior visits to the TV product page** are more likely to click the promotional email.
- **Higher total time spent on the website** correlates with increased engagement.
- **Recency of website visits** (e.g., `days_since_last_visit`) impacts user responsiveness.

#### Model Performance:
- **Incorporating browsing history** significantly enhances prediction accuracy compared to using demographic data alone.
- **Feature engineering**, such as capturing user activity patterns, contributes to model effectiveness.

#### Limitations:
- The model assumes consistent user behavior, which may change due to external factors like market trends or seasonal effects.
- Potential data biases, such as class imbalance, may affect prediction accuracy.
