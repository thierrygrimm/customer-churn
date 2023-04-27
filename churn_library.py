"""
Library of functions to find customers who are likely to churn.

author: Thierry Grimm
date: April 2023
"""

# Imports
import os
import seaborn as sns
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            dataframe: pandas dataframe
    '''
    try:
        # Read the dataframe
        dataframe = pd.read_csv(pth, index_col="Unnamed: 0")
        # Generate Churn Column
        dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        return dataframe
    except FileNotFoundError as err:
        print(f"File at {pth} could not be found")
        raise err


def perform_eda(dataframe):
    '''
    perform eda on dataframe and save figures to images folder
    input:
            dataframe: pandas dataframe
    output:
            None
    '''
    # Histogram of churn
    fig_churn_hist = plt.figure(figsize=(20, 10))
    dataframe['Churn'].hist()
    fig_churn_hist.savefig("images/eda/churn_hist.png")
    plt.clf()

    #  Histogram of customer age
    fig_age_hist = plt.figure(figsize=(20, 10))
    dataframe['Customer_Age'].hist()
    fig_age_hist.savefig("images/eda/age_hist.png")
    plt.clf()

    # Bar plot with relative frequencies of 'Marital_Status'
    fig_marital_bar = plt.figure(figsize=(20, 10))
    dataframe.Marital_Status.value_counts('normalize').plot(kind='bar')
    fig_marital_bar.savefig("images/eda/marital_bar.png")
    plt.clf()

    # Distribution of 'Total_Trans_Ct' with a kernel density estimate curve
    fig_total_trans_histplot = plt.figure(figsize=(20, 10))
    sns.histplot(dataframe['Total_Trans_Ct'], stat='density', kde=True)
    fig_total_trans_histplot.savefig("images/eda/total_trans_histplot.png")
    plt.clf()

    # Heatmap of the correlation matrix
    fig_heatmap = plt.figure(figsize=(20, 10))
    sns.heatmap(dataframe.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    fig_heatmap.savefig("images/eda/Heatmap.png")
    plt.clf()


def encoder_helper(dataframe, category_lst, response=None):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            dataframe: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: (str) response name [optional argument for naming vars or index y column]

    output:
            dataframe: pandas dataframe with new columns for
    '''
    # Calculate proportion of churn
    for i, cat in enumerate(category_lst):
        mean_churn_grouped = dataframe.groupby(cat).mean()['Churn']

        # Save to new column(s)
        if response is None:
            dataframe[cat +
                      '_Churn'] = dataframe[cat].map(dict(mean_churn_grouped))
        else:
            dataframe[response[i]] = dataframe[cat].map(
                dict(mean_churn_grouped))

    return dataframe


def perform_feature_engineering(dataframe, response=None):
    '''
    input:
              dataframe: pandas dataframe
              response: (str) response name [optional argument for naming vars or index y column]

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # Columns to be used for training
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    # Features
    x_data = dataframe[keep_cols].copy()

    # Target
    y_data = dataframe['Churn']
    if response is not None:
        y_data.rename(columns={"Churn": response}, inplace=True)

    return train_test_split(x_data, y_data, test_size=0.3, random_state=42)


def classification_report_image(y_train_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    (y_train, y_test) = y_train_test
    # Saves random forest classification report as image
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_test, y_test_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig("images/results/random_forest.png", bbox_inches="tight")
    plt.clf()

    # Saves logistic regression classification report as image
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_test, y_test_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig("images/results/logistic_regression.png", bbox_inches="tight")
    plt.clf()

def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''

    # Calculate feature importances
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    # Save to file at output_pth
    plt.savefig(output_pth)
    plt.clf()

def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # Training parameters
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # Model specification: Random Forest
    rfc = RandomForestClassifier(random_state=42)

    # Grid search
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    # Fit the data
    cv_rfc.fit(x_train, y_train)

    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(x_train, y_train)

    # Model specification: Logistic Regression
    # Use a different solver if the default 'lbfgs' fails to converge
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    try:
        # Fit the data
        lrc.fit(x_train, y_train)
    except OSError as err:
        print("Logistic Regression solver failed to converge")
        raise err

    # Save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Make predictions from models
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)
    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # Store classification report as image
    classification_report_image(
        (y_train, y_test),
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    # Store ROC curve
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    plot_roc_curve(
        cv_rfc.best_estimator_,
        x_test,
        y_test,
        ax=axis,
        alpha=0.8)
    plot_roc_curve(lrc, x_test, y_test, ax=axis, alpha=0.8)
    plt.savefig("images/results/roc_curve.png")
    plt.clf()

    # Store feature importances
    x_train_test = x_train.append(x_test)
    feature_importance_plot(
        cv_rfc.best_estimator_,
        x_train_test,
        "images/results/feature_importances.png")


if __name__ == "__main__":
    # Import the dataset
    DATAFRAME = import_data(r"./data/bank_data.csv")

    # Perform exploratory data analysis
    perform_eda(DATAFRAME)

    # Define all categorical variables
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    # Encodes all categorical variables to columns with the respective
    # proportion of churn
    DATAFRAME = encoder_helper(DATAFRAME, cat_columns)

    # Split the dataset
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(DATAFRAME)

    # Train the model and save all results
    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
