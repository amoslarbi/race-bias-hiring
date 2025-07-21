# Racial Bias Detection in Hiring Using XAI

This repository contains a Python implementation for detecting racial bias in hiring decisions using Explainable AI (XAI) techniques, as outlined in *Chapter 5: Detecting Racial Bias in Hiring Using XAI: A Practical Implementation*. The project leverages machine learning models, specifically a Random Forest Classifier, combined with interpretability tools like SHAP and Fairlearn to analyze callback rates and uncover potential biases related to race in hiring outcomes.

## Project Overview

The goal of this project is to investigate whether race disproportionately influences job callback rates by implementing a machine learning pipeline with a focus on fairness and transparency. The dataset includes applicant attributes such as race, gender, years of experience, and resume quality, which are used to predict callback outcomes. The analysis employs exploratory data analysis (EDA), model training, and XAI techniques to quantify and visualize bias, ensuring ethical AI practices.

### Key Features
- **Data Preprocessing**: Cleansing and encoding of features like race, gender, and resume quality for numerical analysis.
- **Exploratory Data Analysis (EDA)**: Visualizes callback rate disparities across racial groups using bar plots and chi-squared tests.
- **Model Training**: Utilizes a Random Forest Classifier to predict callback outcomes, addressing class imbalance with RandomUnderSampler.
- **Bias Detection**: Employs SHAP for feature importance and dependence plots, and Fairlearn for fairness metrics like Demographic Parity Difference (DPD) and Equalized Odds Difference (EOD).
- **Statistical Analysis**: Uses logistic regression to test the statistical significance of race and other features in predicting callbacks.

## Repository Structure

- `data/`: Contains the dataset used for analysis (not included in the repo; users must provide their own dataset matching the described structure).
- `notebooks/`: Jupyter notebooks with the implementation, including data preprocessing, EDA, model training, SHAP analysis, and logistic regression.
- `scripts/`: Python scripts for modularized code, including data processing, model training, and bias evaluation.
- `figures/`: Generated visualizations, such as SHAP summary plots, dependence plots, and confusion matrices.
- `requirements.txt`: Lists required Python packages for running the project.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd race-bias-hiring
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Required packages include:
   - `pandas`
   - `numpy`
   - `scikit-learn`
   - `imblearn`
   - `shap`
   - `fairlearn`
   - `matplotlib`
   - `seaborn`
   - `statsmodels`

4. **Prepare the Dataset**:
   - Place your dataset in the `data/` directory.
   - Ensure it includes columns like `race`, `gender`, `years_experience`, `college_degree`, `years_college`, `computer_skills`, `special_skills`, `resume_quality`, and `received_callback` as described in the documentation.

## Usage

1. **Run the Jupyter Notebooks**:
   - Open the notebooks in the `notebooks/` directory using Jupyter Lab or Jupyter Notebook.
   - Execute the cells sequentially to preprocess data, perform EDA, train the model, and generate visualizations.

2. **Run Scripts**:
   - Execute the main script for end-to-end analysis:
     ```bash
     python scripts/main.py
     ```
   - Individual scripts for specific tasks (e.g., data preprocessing, model training) can be run separately.

3. **View Results**:
   - Visualizations (e.g., SHAP plots, confusion matrices) are saved in the `figures/` directory.
   - Check the console output or notebook cells for metrics like accuracy, precision, recall, SHAP values, and fairness scores.

## Key Findings

- **Callback Rate Disparities**: EDA reveals a 9% callback rate for White applicants compared to 6% for Black applicants, suggesting potential bias.
- **Model Performance**: The Random Forest model achieves 59.4% accuracy on a balanced dataset, with higher recall (67.4% vs. 54.2%) and precision (62.0% vs. 50.0%) for White applicants, indicating differential performance.
- **SHAP Analysis**: Race has a significant impact (mean absolute SHAP value of 0.06), with White applicants receiving a positive boost (+0.038) and Black applicants facing a negative impact (-0.044) in callback predictions.
- **Logistic Regression**: The race coefficient (0.3638) is not statistically significant (p-value = 0.1169), suggesting limitations in detecting bias with linear models.
- **Fairness Metrics**: SHAP and Fairlearn analyses confirm bias, with race influencing predictions beyond qualifications, necessitating fairness interventions.

## Limitations

- **Dataset Size**: The small sample size (331 observations after balancing) limits statistical power, particularly for logistic regression.
- **Feature Simplification**: Binary encoding of race and gender may overlook intersectional effects (e.g., Black female applicants).
- **Model Constraints**: The Random Forest model captures non-linear patterns but may overfit, and fairness constraints were not applied during training.

## Recommendations

- **Enhance Dataset**: Collect more data to improve statistical power and generalizability.
- **Fairness Interventions**: Implement fairness-aware algorithms (e.g., reweighing, fairness constraints) or remove race as a feature to mitigate bias.
- **Model Tuning**: Experiment with alternative models (e.g., XGBoost) and hyperparameter tuning to balance accuracy and fairness.
- **Further Analysis**: Investigate potential proxy features (e.g., resume quality) that may indirectly encode racial bias.

## Contributing

Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add feature-name'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## Acknowledgments

- The implementation is based on *My Master Thesis: Detecting Racial Bias in Hiring Using XAI: A Practical Implementation*.
- Thanks to the open-source communities behind `scikit-learn`, `SHAP`, `Fairlearn`, and other libraries used in this project.
