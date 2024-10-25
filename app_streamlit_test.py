#------------------------------------------------------------------------------------------------------------------------------------------------------
#''' Library Imports '''
import numpy as np
import pandas as pd
pd.options.display.max_columns = None
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve,  precision_recall_curve, auc, log_loss
from sklearn.utils import resample

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import gower
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA, TruncatedSVD
from kmodes.kmodes import KModes

import streamlit as st
#------------------------------------------------------------------------------------------------------------------------------------------------------
#''' Data Imports '''
path_dataset = './2024-10-21 - data_cfasf_membership_kept_col.csv'
data_imported = pd.read_csv(path_dataset, encoding='latin-1')
data = data_imported
#------------------------------------------------------------------------------------------------------------------------------------------------------
#''' Data Preprocessing '''
# Separating features into types
cat_seniority = ['Other', 'Entry-Level', 'Early/Mid-Level', 'Mid-Level', 'Senior-Level', 'Executive-Level', 'C-Suite']
col_categorical_ordinal = ['seniority']
col_categorical_nominal = ['gender', 'type_cfai_membership', 'employment_status', 'employer_type'] 
col_numeric = ['age', 'is_in_france', 'year_joined', 'duration_membership_years', 'duration_program_years', 'is_on_professional_leave']
col_categorical = col_categorical_nominal + col_categorical_ordinal

# Define encoders for different features
preprocessor = ColumnTransformer(
    transformers=[
        ('ordinal', OrdinalEncoder(categories=[cat_seniority]), col_categorical_ordinal),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), col_categorical_nominal),  
        ('scaler', StandardScaler(), col_numeric)   # Removed StandardScaler for numeric features
    ],
    remainder='passthrough'  # Keep columns not specified in transformers
)
print(f"Number of selected columns is {len(col_categorical_nominal) + len(col_categorical_ordinal) + len(col_numeric)}")

# Define features and target
kept_columns = col_categorical_nominal + col_categorical_ordinal + col_numeric
X = data[kept_columns]
y = data['churned']  # Target variable
# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#------------------------------------------------------------------------------------------------------------------------------------------------------
#''' Model Development '''
# Logistic Regression & Random Forest Pipelines
pipeline_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=10000, class_weight='balanced')) #LogisticRegression(random_state=42, class_weight='balanced')
])
model_lr = pipeline_lr
pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced')) 
])
model_rf = pipeline_rf

# Training models
model_lr.fit(X_train, y_train)
classifier_lr = model_lr.named_steps['classifier']
# print("Parameters found: ", classifier_lr.get_params())
model_rf.fit(X_train, y_train)
classifier_rf = model_rf.named_steps['classifier']
# print("Parameters found: ", classifier_rf.get_params())

# Predictions
def get_model_predictions(model, X_test):
    print(model.named_steps['classifier'].get_params()) 
    y_pred_lr = model.predict(X_test)
    y_pred_lr_churn_proba = model.predict_proba(X_test)[:, 1]
    y_pred_lr_both_proba = model.predict_proba(X_test)  
    return y_pred_lr, y_pred_lr_churn_proba, y_pred_lr_both_proba

y_pred_lr, y_pred_lr_churn_proba, y_pred_lr_both_proba = get_model_predictions(model_lr, X_test)
y_pred_rf, y_pred_rf_churn_proba, y_pred_rf_both_proba = get_model_predictions(model_rf, X_test)

# Evaluations
def evaluate_model(model, y_true, y_pred, y_pred_proba):
    print("Model Params:\t", model.named_steps['classifier'].get_params())
    acc = accuracy_score(y_true, y_pred)
    print("Accuracy:\t", acc)
    print("AUC-ROC Score:\t", roc_auc_score(y_true, y_pred_proba))
    clf_report = classification_report(y_true, y_pred)
    print("Classification Report:\n", clf_report)
    # Display the confusion matrix
    print(confusion_matrix(y_true, y_pred))
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    print("\n")
    return acc, clf_report

acc_lr, report_lr = evaluate_model(model_lr, y_test, y_pred_lr, y_pred_lr_churn_proba)
acc_rf, report_rf = evaluate_model(model_rf, y_test, y_pred_rf, y_pred_rf_churn_proba)
#------------------------------------------------------------------------------------------------------------------------------------------------------
#''' Descriptive analytics '''
def descriptive_analytics(df):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Age distribution
    sns.histplot(df['age'], bins=20, ax=ax[0])
    ax[0].set_title('Age Distribution')
    
    # Churn distribution
    sns.countplot(x='churned', data=df, ax=ax[1])
    ax[1].set_title('Churned vs Non-Churned')
    
    return fig
#------------------------------------------------------------------------------------------------------------------------------------------------------
# Streamlit app
def main():
    df = data
    st.title("Membership Churn Prediction Dashboard")

    # Sidebar options
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Select an option", ["Descriptive Analytics", "Model Results"])

    # Descriptive Analytics
    if options == "Descriptive Analytics":
        st.header("Descriptive Analytics")
        st.pyplot(descriptive_analytics(df))

    # Predictive Analytics (Model Results)
    elif options == "Model Results":
        st.header("Model Evaluation")

        st.subheader("Logistic Regression Results")
        st.text(f"Accuracy: {acc_lr}")
        st.text(report_lr)

        st.subheader("Random Forest Results")
        st.text(f"Accuracy: {acc_rf}")
        st.text(report_rf)

        # Option to run on user inputs
        if st.sidebar.checkbox("Predict Churn for a New Member"):
            age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
            seniority = st.sidebar.slider("Seniority", min_value=0, max_value=40, value=10)
            employment_status = st.sidebar.selectbox("Employment Status", options=df['employment_status'].unique())
            gender = st.sidebar.selectbox("Gender", options=df['gender'].unique())

            new_member = pd.DataFrame({
                'age': [age],
                'seniority': [seniority],
                'employment_status': [employment_status],
                'gender': [gender]
            })

            # new_member_pred = model_lr.predict(new_member)
            # st.write(f"The model predicts that this member will {'churn' if new_member_pred[0] == 1 else 'not churn'}")

if __name__ == '__main__':
    main()