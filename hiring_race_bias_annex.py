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
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler

# Load the dataset
data = pd.read_csv('Which_resume_attributes_drive_job_callbacks___Race_and_gender_under_study.csv')

# First column is dropped becasue it is an index, hence, an irrelevant identifier
data = data.drop(data.columns[0], axis=1)

# Handle missing values (replace empty strings with NaN)
data.replace('', np.nan, inplace=True)

# Drop rows with missing critical columns
data.dropna(subset=['race', 'gender', 'received_callback'], inplace=True)

# Encode categorical features
le = LabelEncoder()
categorical_cols = ['race', 'gender', 'college_degree', 'resume_quality', 'job_industry']
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Define features and target
features = ['race', 'gender', 'years_experience', 'college_degree', 'years_college', 'computer_skills', 'special_skills', 'resume_quality']
target = 'received_callback'

X = data[features]
y = data[target]

# Split the dataset into training and testing sets (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Calculate callback rates by race
callback_by_race = data.groupby('race')['received_callback'].mean()
print("\nCallback Rates by Race:")
print(callback_by_race)  # 0 = black, 1 = white after encoding

# Visualize callback rates
plt.bar(['Black (0)', 'White (1)'], callback_by_race)
plt.title('Callback Rates by Race')
plt.ylabel('Callback Rate')
plt.show()

# Chi-squared test for independence between race and callback
contingency_table = pd.crosstab(data['race'], data['received_callback'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-squared Test for Race: chi2={chi2:.4f}, p-value={p:.4f}")

# Train Random Forest model before handling imbalance
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model
accuracy = rf_model.score(X_test, y_test)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Generate predictions for confusion matrix before handling imbalance
y_pred = rf_model.predict(X_test)

# Compute and display confusion matrix before handling imbalance
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Callback (0)', 'Callback (1)'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Random Forest Model (Imbalanced)")
plt.show()

# Subset predictions by race
black_mask = X_test['race'] == 0
white_mask = X_test['race'] == 1

# Confusion matrices by race before handling imbalance
cm_black = confusion_matrix(y_test[black_mask], y_pred[black_mask])
cm_white = confusion_matrix(y_test[white_mask], y_pred[white_mask])

disp_black = ConfusionMatrixDisplay(confusion_matrix=cm_black, display_labels=['No Callback (0)', 'Callback (1)'])
disp_white = ConfusionMatrixDisplay(confusion_matrix=cm_white, display_labels=['No Callback (0)', 'Callback (1)'])

disp_black.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: Black Applicants (Imbalanced)")
plt.show()

disp_white.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: White Applicants (Imbalanced)")
plt.show()

# Apply RandomUnderSampler to handle imbalance
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Split the dataset into training and testing sets (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Train Random Forest model after handling imbalance
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Evaluate model
accuracy = rf_model.score(X_test, y_test)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Generate predictions for confusion matrix after handling imbalance
y_pred = rf_model.predict(X_test)

# Compute and display confusion matrix after handling imbalance
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Callback (0)', 'Callback (1)'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Random Forest Model (Balanced)")
plt.show()

# Subset predictions by race
black_mask = X_test['race'] == 0
white_mask = X_test['race'] == 1

# Confusion matrices by race after handling imbalance
cm_black = confusion_matrix(y_test[black_mask], y_pred[black_mask])
cm_white = confusion_matrix(y_test[white_mask], y_pred[white_mask])

disp_black = ConfusionMatrixDisplay(confusion_matrix=cm_black, display_labels=['No Callback (0)', 'Callback (1)'])
disp_white = ConfusionMatrixDisplay(confusion_matrix=cm_white, display_labels=['No Callback (0)', 'Callback (1)'])

disp_black.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: Black Applicants (Balanced)")
plt.show()

disp_white.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: White Applicants (Balanced)")
plt.show()

# Feature importance from Random Forest
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
print("\nFeature Importance:")
print(feature_importance_df.sort_values(by='Importance', ascending=False))

# Select top 8 most important features
top_features = feature_importance_df.sort_values(by='Importance', ascending=False).head(8)['Feature'].tolist()
X_train_top = X_train[top_features]
X_test_top = X_test[top_features]

# Retrain Random Forest model with top features
rf_model_top = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_top.fit(X_train_top, y_train)

# Compute SHAP values for the top features
explainer = shap.TreeExplainer(rf_model_top)
shap_values_top = explainer.shap_values(X_test_top)

# Correlation matrix
corr_matrix = X_train.corr(method='pearson')
print("\nCorrelation Matrix:")
print(corr_matrix)

# VIF analysis
X_vif = X_train.copy()
X_vif = sm.add_constant(X_vif)
vif_data = pd.DataFrame()
vif_data['Feature'] = X_vif.columns
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print("\nVIF Analysis:")
print(vif_data)

# Summary plot for top features (positive class)
shap.summary_plot(shap_values_top[:,:,1], X_test_top, plot_type="bar")
plt.title("SHAP Feature Importance for Callback Prediction")
plt.show()

# Beeswarm plot for top features (positive class)
shap.summary_plot(shap_values_top[:,:,1], X_test_top)
plt.title("SHAP Beeswarm Plot for Callback Prediction")
plt.show()

# Dependence plot for race (if race is in top features)
if 'race' in top_features:
    shap.dependence_plot('race', shap_values_top[:,:,1], X_test_top, show=False)
    plt.title("SHAP Dependence Plot for Race")
    plt.show()

# Average SHAP value for race by race group (if race is in top features)
if 'race' in top_features:
    shap_race = pd.DataFrame({'SHAP': shap_values_top[:,:,1][:, X_test_top.columns.get_loc('race')], 'Race': X_test_top['race']})
    shap_by_race = shap_race.groupby('Race')['SHAP'].mean()
    print("\nAverage SHAP Values for Race:")
    print(shap_by_race)  # 0 = black, 1 = white

# Logistic regression to assess statistical significance of race
X_train_logit = sm.add_constant(X_train_top)  # Add intercept
logit_model = sm.Logit(y_train, X_train_logit).fit()
print("\nLogistic Regression Summary:")
print(logit_model.summary())

# Focus on race coefficient (if race is in top features)
if 'race' in top_features:
    race_coef = logit_model.params['race']
    race_pvalue = logit_model.pvalues['race']
    print(f"\nRace Coefficient: {race_coef:.4f}, p-value: {race_pvalue:.4f}")