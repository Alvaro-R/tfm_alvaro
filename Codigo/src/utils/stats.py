import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import VarianceThreshold


def calculate_vif(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the VIF for each feature in a pandas DataFrame.

    Args:
    dataframe (pandas.DataFrame): The input features.

    Returns:
    pandas.DataFrame: The VIF for each feature.
    """
    # CREATE AN EMPTY DATAFRAME FOR VIF VALUES
    vif = pd.DataFrame()
    # ADD VARIABLE NAMES TO THE DATAFRAME
    vif["Features"] = dataframe.columns
    # CALCULATE VIF FOR EACH VARIABLE
    vif["VIF"] = [
        variance_inflation_factor(dataframe.values, i)
        for i in range(dataframe.shape[1])
    ]
    return vif


def vif_filter(dataframe: pd.DataFrame, treshold=5.0) -> pd.DataFrame:
    """
    Remove features with high VIFs iteratively until all VIFs are below the threshold.

    Args:
        dataframe (pandas.DataFrame): The input features.
        treshold (float, optional): The threshold VIF. Default is 5.0.

    Returns:
        pandas.DataFrame: The input features with high VIFs removed.
    """
    vif = calculate_vif(dataframe)
    while vif["VIF"].max() > treshold:
        max_vif_feature = vif.loc[vif["VIF"].idxmax(), "Features"]
        print(f"Removing {max_vif_feature} with VIF {vif['VIF'].max()}")
        dataframe = dataframe.drop(max_vif_feature, axis=1)
        vif = calculate_vif(dataframe)
    return dataframe


def pearson_corr_filter(dataframe: pd.DataFrame, threshold=0.8) -> list:
    """
    Return list of high correlted variables from a dataframe.

    Args:
        dataframe (pandas.DataFrame): The input dataframe with numerical
        columns.
        threshold (float, optional): The correlation threshold to use.
        Any pair of columns with a correlation coefficient greater than
        or equal to the threshold will be considered correlated.
        Default is 0.8.

    Returns:
        list: The list of high correlated columns.

    """
    # CALCULATE CORRELATION MATRIX WITH ABSOLUTE VALUES
    corr_matrix = dataframe.corr().abs()

    # GET UPPER TRIANGLE OF CORRELATION MATRIX
    corr_upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # FILTER COLUMNS WITH CORRELATION GREATER THAN THRESHOLD
    drop_columns = [
        column for column in corr_upper.columns if any(corr_upper[column] > threshold)
    ]

    # RETURN DATAFRAME WITHOUT CORRELATED COLUMNS
    return drop_columns


def low_variance_filter(dataframe: pd.DataFrame, threshold=0) -> pd.DataFrame:
    """
    Return list of low variance variables from a dataframe.

    Args:
        dataframe (pd.DataFrame): The input dataframe with numerical
        columns.
        threshold (int, optional): The variance threshold to use. Any column with a
        variance less than or equal to the threshold will be considered low
        variance. Default is 0.

    Returns:
        list: The list of low variance columns.
    """
    # DEFINE SELECTOR THRESHOLD
    selector = VarianceThreshold(threshold=threshold)
    # FIT IT TO DATA
    selector.fit_transform(dataframe)
    # RETURN COLUMNS WITH LOW VARIANCE
    return list(dataframe.columns[selector.get_support()])
