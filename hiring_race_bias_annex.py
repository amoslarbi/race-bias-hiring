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
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler

# Load the dataset
data = pd.read_csv('Which_resume_attributes_drive_job_callbacks___Race_and_gender_under_study.csv')

# First column is dropped becasue it is an index, hence, an irrelevant identifier
data = data.drop(data.columns[0], axis=1)

# Handle missing values (replace empty strings with NaN)
data.replace('', np.nan, inplace=True)

# Drop rows with missing critical columns (e.g., race, received_callback)
data.dropna(subset=['race', 'received_callback'], inplace=True)

# Encode categorical variables
le = LabelEncoder()
categorical_cols = ['race', 'job_type', 'job_industry', 'job_ownership', 'resume_quality', 'gender']
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Convert binary columns to integers
binary_cols = ['job_fed_contractor', 'job_equal_opp_employer', 'job_req_any', 'job_req_communication',
               'job_req_education', 'job_req_computer', 'job_req_organization', 'college_degree', 
               'honors', 'worked_during_school', 'computer_skills', 'special_skills', 'volunteer', 
               'military', 'employment_holes', 'has_email_address']
for col in binary_cols:
    data[col] = data[col].astype(int)

features = ['race', 'gender', 'years_experience', 'college_degree', 'years_college', 'computer_skills', 'special_skills', 'resume_quality']
target = 'received_callback'

# Define features and target
# X = data.drop(columns=['received_callback', 'firstname', 'job_ad_id', 'job_city', 'job_req_min_experience', 
#                        'job_req_school'])  # Drop non-predictive or redundant columns
# y = data['received_callback']

X = data[features]
y = data[target]

# Apply RandomUnderSampler to handle imbalance
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Split the dataset into training and testing sets (70/30 split)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

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

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Evaluate model
accuracy = rf_model.score(X_test, y_test)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Generate predictions for confusion matrix
y_pred = rf_model.predict(X_test)

# Compute and display confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Callback (0)', 'Callback (1)'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Random Forest Model")
plt.show()

# Subset predictions by race
black_mask = X_test['race'] == 0
white_mask = X_test['race'] == 1

# Confusion matrices by race
cm_black = confusion_matrix(y_test[black_mask], y_pred[black_mask])
cm_white = confusion_matrix(y_test[white_mask], y_pred[white_mask])

disp_black = ConfusionMatrixDisplay(confusion_matrix=cm_black, display_labels=['No Callback (0)', 'Callback (1)'])
disp_white = ConfusionMatrixDisplay(confusion_matrix=cm_white, display_labels=['No Callback (0)', 'Callback (1)'])

disp_black.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Random Forest Model Black")
plt.show()

disp_white.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Random Forest Model White")
plt.show()

# Print confusion matrix details
tn, fp, fn, tp = cm.ravel()
print("\nConfusion Matrix Details:")
print(f"True Negatives (TN): {tn} (Predicted No Callback, Actual No Callback)")
print(f"False Positives (FP): {fp} (Predicted Callback, Actual No Callback)")
print(f"False Negatives (FN): {fn} (Predicted No Callback, Actual Callback)")
print(f"True Positives (TP): {tp} (Predicted Callback, Actual Callback)")

# Feature importance from Random Forest
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
print("\nFeature Importance:")
print(feature_importance_df.sort_values(by='Importance', ascending=False))

# Select top 10 most important features
top_features = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)['Feature'].tolist()
X_train_top = X_train[top_features]
X_test_top = X_test[top_features]

# Retrain Random Forest model with top features
rf_model_top = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_top.fit(X_train_top, y_train)

# Compute SHAP values for the top features
explainer = shap.TreeExplainer(rf_model_top)
shap_values_top = explainer.shap_values(X_test_top)

# Summary plot for top features (positive class)
shap.summary_plot(shap_values_top[:,:,1], X_test_top, plot_type="bar")
plt.title("SHAP Feature Importance for Callback Prediction (Top 10 Features)")
plt.show()

# Beeswarm plot for top features (positive class)
shap.summary_plot(shap_values_top[:,:,1], X_test_top)
plt.title("SHAP Beeswarm Plot for Callback Prediction (Top 10 Features)")
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







# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from sklearn.preprocessing import LabelEncoder
# from imblearn.under_sampling import RandomUnderSampler

# # Load the dataset
# data = pd.read_csv('Which_resume_attributes_drive_job_callbacks___Race_and_gender_under_study.csv')

# # Encode categorical features
# categorical_columns = ['race', 'gender', 'college_degree', 'resume_quality']
# data_encoded = data.copy()
# for column in categorical_columns:
#     data_encoded[column] = data_encoded[column].astype('category').cat.codes

# # Select features and target variable
# features = ['race', 'gender', 'years_experience', 'college_degree', 'years_college', 'computer_skills', 'special_skills', 'resume_quality']
# target = 'received_callback'

# X = data_encoded[features]
# y = data_encoded[target]

# # Apply RandomUnderSampler to handle imbalance
# rus = RandomUnderSampler(random_state=42)
# X_resampled, y_resampled = rus.fit_resample(X, y)

# # Split data into training and testing
# X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# # Train Random Forest Classifier
# clf = RandomForestClassifier(random_state=42)
# clf.fit(X_train, y_train)

# # Predict on the test set
# y_pred = clf.predict(X_test)

# # Generate confusion matrix
# conf_matrix = confusion_matrix(y_test, y_pred)

# # Plot confusion matrix
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix: Callbacks by Race (Balanced Data)')
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.show()