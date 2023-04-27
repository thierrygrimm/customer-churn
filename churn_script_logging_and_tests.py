"""
Unit tests for churn_library

author: Thierry Grimm
date: April 2023
"""

import os
import logging
import pytest
import churn_library as cls


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture(scope="module", name="import_data")
def fix_import_data():
    '''
    Pytest helper function for import_data
    '''
    return cls.import_data


def test_import(import_data):
    '''
    test data import
    '''
    try:
        dataframe = import_data("./data/bank_data.csv")
        logging.info("SUCCESS: Testing import_data")
    except FileNotFoundError as err:
        logging.error("ERROR: Testing import_data: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


@pytest.fixture(scope="module", name="perform_eda")
def fix_perform_eda():
    '''
    Pytest helper function for perform_eda
    '''
    return cls.perform_eda


def test_eda(perform_eda, import_data):
    '''
    test perform eda function
    '''
    try:
        dataframe = import_data("./data/bank_data.csv")
        perform_eda(dataframe)
        logging.info("SUCCESS: Testing perform_eda")
    except BaseException:
        logging.error(
            "ERROR: Testing perform_eda: Exploratory data analysis was not completed")
        raise

    try:
        # Check if plots were created
        assert os.path.exists("images/eda/churn_hist.png")
        assert os.path.exists("images/eda/age_hist.png")
        assert os.path.exists("images/eda/marital_bar.png")
        assert os.path.exists("images/eda/total_trans_histplot.png")
        assert os.path.exists("images/eda/Heatmap.png")
    except AssertionError as err:
        logging.error(
            "ERROR: Testing perform_eda: The images were not written")
        raise err


@pytest.fixture(scope="module", name="encoder_helper")
def fix_encoder_helper():
    '''
    Pytest helper function for encoder_helper
    '''
    return cls.encoder_helper


def test_encoder_helper(encoder_helper, import_data):
    '''
    test encoder helper
    '''
    try:
        dataframe = import_data("./data/bank_data.csv")
        category_lst = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category']
        encoder_helper(dataframe, category_lst, response=None)
        logging.info("SUCCESS: Testing encoder_helper")
    except BaseException:
        logging.error(
            "ERROR: Testing encoder_helper: Could not encode the columns")
        raise

    try:
        # Check if there was at least one category supplied
        assert len(category_lst) > 0
    except AssertionError as err:
        logging.info("INFO: No categories were supplied")
        raise err

    try:
        # Check if the category that was supplied is present in the data
        for category in category_lst:
            current = category
            assert category in dataframe.columns
    except AssertionError as err:
        logging.error(
            "ERROR: The category %d was supplied, but not found in the dataframe",
            current)
        raise err


@pytest.fixture(scope="module", name="perform_feature_engineering")
def fix_perform_feature_engineering():
    '''
    Pytest helper function for perform_feature_engineering
    '''
    return cls.perform_feature_engineering


def test_perform_feature_engineering(
        perform_feature_engineering,
        encoder_helper,
        import_data):
    '''
    test perform_feature_engineering
    '''
    try:
        dataframe = import_data("./data/bank_data.csv")
        cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category']
        dataframe = encoder_helper(dataframe, cat_columns)
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            dataframe, response=None)
        logging.info("SUCCESS: Testing perform_feature_engineering")
    except BaseException:
        logging.error(
            "ERROR: Testing perform_feature_engineering: Could not split the dataset")
        logging.error(
            "ERROR: Testing perform_feature_engineering: Columns do not match")
        raise

    try:
        # Check if the sizes match
        assert (len(x_train) + len(x_test)) == len(dataframe)
        assert (len(y_train) + len(y_test)) == len(dataframe)
    except AssertionError as err:
        logging.error("ERROR: Some samples were lost during the split")
        raise err

    try:
        # Check if all arrays contain entries
        assert len(x_train) > 0
        assert len(x_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
    except AssertionError as err:
        logging.warning("WARNING: One set contains no samples")
        raise err


@pytest.fixture(scope="module", name="train_models")
def fix_train_models():
    '''
    Pytest helper function for train_models
    '''
    return cls.train_models


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_train_models(
        train_models,
        perform_feature_engineering,
        encoder_helper,
        import_data):
    '''
    test train_models
    '''
    try:
        dataframe = import_data("./data/bank_data.csv")
        cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category']
        dataframe = encoder_helper(dataframe, cat_columns)
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            dataframe)
        train_models(x_train, x_test, y_train, y_test)
        logging.info("SUCCESS: Testing train_models")
    except OSError as err:
        logging.error(
            "ERROR: Testing train_models: Logistic Regression solver failed to converge")
        raise err
    except BaseException:
        logging.error("ERROR: Testing train_models: Could not train model")
        raise

    try:
        # Check if model was stored
        assert os.path.exists("./models/rfc_model.pkl")
        assert os.path.exists("./models/logistic_model.pkl")
    except AssertionError as err:
        logging.error("ERROR: Models were not saved")
        raise err

    try:
        # Check if classification reports were saved as images
        assert os.path.exists("images/results/random_forest.png")
        assert os.path.exists("images/results/logistic_regression.png")
    except AssertionError as err:
        logging.error("ERROR: Classification reports were not generated")
        raise err

    try:
        # Check if ROC curve was saved
        assert os.path.exists("images/results/roc_curve.png")
    except AssertionError as err:
        logging.error("ERROR: ROC curve was not saved")
        raise err

    try:
        # Check if the feature importances plot was saved
        assert os.path.exists("images/results/feature_importances.png")
    except AssertionError as err:
        logging.error("ERROR: Feature importances were not saved")
        raise err


if __name__ == "__main__":
    # Execute pytest on run
    test = pytest.main()
