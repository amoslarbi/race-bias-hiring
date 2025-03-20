import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import shap
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from imblearn.under_sampling import RandomUnderSampler

# Load the dataset
data = pd.read_csv('Which_resume_attributes_drive_job_callbacks___Race_and_gender_under_study.csv')

# Drop the index column
data = data.drop(data.columns[0], axis=1)

# Handle missing values
data.replace('', np.nan, inplace=True)
data.dropna(subset=['race', 'gender', 'received_callback'], inplace=True)  # Critical columns

# Encode categorical variables
le = LabelEncoder()
categorical_cols = ['race', 'gender', 'job_industry', 'job_type', 'job_ownership', 'resume_quality']
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Ensure binary columns are integers
binary_cols = ['job_fed_contractor', 'job_equal_opp_employer', 'job_req_any', 'job_req_communication',
               'job_req_education', 'job_req_computer', 'job_req_organization', 'college_degree',
               'honors', 'worked_during_school', 'computer_skills', 'special_skills', 'volunteer',
               'military', 'employment_holes', 'has_email_address']
for col in binary_cols:
    data[col] = data[col].astype(int)

# Select features and target
features = ['race', 'gender', 'years_experience', 'college_degree', 'years_college', 'computer_skills',
            'special_skills', 'resume_quality', 'job_industry', 'job_type', 'job_ownership']
X = data[features]
y = data['received_callback']

# Calculate callback rates by race and gender
callback_by_race = data.groupby('race')['received_callback'].mean()
callback_by_gender = data.groupby('gender')['received_callback'].mean()
print("\nCallback Rates by Race:")
print(callback_by_race)  # 0 = black, 1 = white
print("\nCallback Rates by Gender:")
print(callback_by_gender)  # 0 = female, 1 = male

# Visualize callback rates
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(['Black (0)', 'White (1)'], callback_by_race)
plt.title('Callback Rates by Race')
plt.ylabel('Callback Rate')
plt.subplot(1, 2, 2)
plt.bar(['Female (0)', 'Male (1)'], callback_by_gender)
plt.title('Callback Rates by Gender')
plt.show()

# Chi-squared tests
contingency_race = pd.crosstab(data['race'], data['received_callback'])
chi2_race, p_race, _, _ = chi2_contingency(contingency_race)
print(f"\nChi-squared Test for Race: chi2={chi2_race:.4f}, p-value={p_race:.4f}")

contingency_gender = pd.crosstab(data['gender'], data['received_callback'])
chi2_gender, p_gender, _, _ = chi2_contingency(contingency_gender)
print(f"Chi-squared Test for Gender: chi2={chi2_gender:.4f}, p-value={p_gender:.4f}")

# Handle class imbalance with RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Split the dataset (70/30)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model
accuracy = rf_model.score(X_test, y_test)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Confusion matrix
y_pred = rf_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Callback (0)', 'Callback (1)'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Random Forest Model (Balanced)")
plt.show()

# Subset predictions by race and gender
black_mask = X_test['race'] == 0
white_mask = X_test['race'] == 1
female_mask = X_test['gender'] == 0
male_mask = X_test['gender'] == 1

cm_black = confusion_matrix(y_test[black_mask], y_pred[black_mask])
cm_white = confusion_matrix(y_test[white_mask], y_pred[white_mask])
cm_female = confusion_matrix(y_test[female_mask], y_pred[female_mask])
cm_male = confusion_matrix(y_test[male_mask], y_pred[male_mask])

disp_black = ConfusionMatrixDisplay(confusion_matrix=cm_black, display_labels=['No Callback (0)', 'Callback (1)'])
disp_white = ConfusionMatrixDisplay(confusion_matrix=cm_white, display_labels=['No Callback (0)', 'Callback (1)'])
disp_female = ConfusionMatrixDisplay(confusion_matrix=cm_female, display_labels=['No Callback (0)', 'Callback (1)'])
disp_male = ConfusionMatrixDisplay(confusion_matrix=cm_male, display_labels=['No Callback (0)', 'Callback (1)'])

plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
disp_black.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: Black Applicants")
plt.subplot(2, 2, 2)
disp_white.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: White Applicants")
plt.subplot(2, 2, 3)
disp_female.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: Female Applicants")
plt.subplot(2, 2, 4)
disp_male.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: Male Applicants")
plt.tight_layout()
plt.show()

# Print confusion matrix details
tn, fp, fn, tp = cm.ravel()
print("\nOverall Confusion Matrix Details:")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")

# Feature importance
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
print("\nFeature Importance:")
print(feature_importance_df.sort_values(by='Importance', ascending=False))

# SHAP analysis
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Summary plot (positive class)
shap.summary_plot(shap_values[1], X_test, plot_type="bar")
plt.title("SHAP Feature Importance for Callback Prediction")
plt.show()

# Beeswarm plot
shap.summary_plot(shap_values[1], X_test)
plt.title("SHAP Beeswarm Plot for Callback Prediction")
plt.show()

# Dependence plots for race and gender
for feature in ['race', 'gender']:
    shap.dependence_plot(feature, shap_values[1], X_test, show=False)
    plt.title(f"SHAP Dependence Plot for {feature.capitalize()}")
    plt.show()

# Average SHAP values by race and gender
shap_race = pd.DataFrame({'SHAP': shap_values[1][:, X_test.columns.get_loc('race')], 'Race': X_test['race']})
shap_gender = pd.DataFrame({'SHAP': shap_values[1][:, X_test.columns.get_loc('gender')], 'Gender': X_test['gender']})
print("\nAverage SHAP Values by Race:")
print(shap_race.groupby('Race')['SHAP'].mean())
print("\nAverage SHAP Values by Gender:")
print(shap_gender.groupby('Gender')['SHAP'].mean())

# Logistic regression for statistical significance
X_train_logit = sm.add_constant(X_train)
logit_model = sm.Logit(y_train, X_train_logit).fit()
print("\nLogistic Regression Summary:")
print(logit_model.summary())

# Focus on race and gender coefficients
for feature in ['race', 'gender']:
    coef = logit_model.params[feature]
    pvalue = logit_model.pvalues[feature]
    print(f"\n{feature.capitalize()} Coefficient: {coef:.4f}, p-value: {pvalue:.4f}")