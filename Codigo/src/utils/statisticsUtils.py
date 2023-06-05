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


def pearson_corr_filter(
    dataframe: pd.DataFrame, threshold=0.8, target=pd.Series
) -> list:
    """
    Return list of high correlted variables from a dataframe.

    Args:
        dataframe (pandas.DataFrame): The input dataframe with numerical
        columns.
        threshold (float, optional): The correlation threshold to use.
        Any pair of columns with a correlation coefficient greater than
        or equal to the threshold will be considered correlated.
        Default is 0.8.
        target (pandas.Series, optional): The target column. Default is None.

    Returns:
        list: The list of high correlated columns.

    """
    # CALCULATE CORRELATION MATRIX WITH ABSOLUTE VALUES
    corr_matrix = dataframe.corr().abs()

    # GET UPPER TRIANGLE OF CORRELATION MATRIX
    corr_upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # GET PAIRS OF CORRELATED VARIABLES
    corr_pairs = corr_upper.unstack().sort_values(ascending=False)

    # GET CORRELATED VARIABLES ABOVE THRESHOLD
    corr_pairs = corr_pairs[(corr_pairs > threshold)].reset_index()

    dropping_variables = []
    # FOR EACH PAIR OF CORRELATED VARIABLES, ADD TO LIST THE ONE
    # LESS CORRELATED WITH THE TARGET
    for i in range(len(corr_pairs)):
        # GET VARIABLES NAMES
        var1 = corr_pairs.iloc[i, 0]
        var2 = corr_pairs.iloc[i, 1]

        # CONCACTENATE VARIABLES WITH TARGET
        df = pd.concat([dataframe[var1], dataframe[var2], target], axis=1)

        # CALCULATE CORRELATION COEFFICIENTS
        corr1 = df.corr(method="kendall").iloc[0, 2]
        corr2 = df.corr(method="kendall").iloc[1, 2]

        # ADD TO LIST THE VARIABLE LESS CORRELATED WITH THE TARGET
        if corr1 < corr2:
            dropping_variables.append(var1)
        else:
            dropping_variables.append(var2)

    # RETURN LIST OF VARIABLES TO DROP WITHOUT DUPLICATES
    return list(set(dropping_variables))


def spearman_corr_filter(
    dataframe: pd.DataFrame, threshold=0.8, target=pd.Series
) -> list:
    """
    Filter variables based on Spearman correlation with a threshold.

    Args:
        dataframe (pd.DataFrame): The input dataframe containing the
        variables.
        threshold (float): The correlation threshold (default: 0.8).
        target (pd.Series): The target variable used for correlation
        comparisons (default: None).

    Returns:
        List[str]: A list of variables to be dropped based on high correlation.
    """
    # CALCULATE CORRELATION MATRIX WITH ABSOLUTE VALUES
    corr_matrix = dataframe.corr(method="spearman").abs()

    # GET UPPER TRIANGLE OF CORRELATION MATRIX
    corr_upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # GET PAIRS OF CORRELATED VARIABLES
    corr_pairs = corr_upper.unstack().sort_values(ascending=False)

    # GET CORRELATED VARIABLES ABOVE THRESHOLD
    corr_pairs = corr_pairs[(corr_pairs > threshold)].reset_index()

    dropping_variables = []
    # FOR EACH PAIR OF CORRELATED VARIABLES, ADD TO LIST
    # THE ONE LESS CORRELATED WITH THE TARGET
    for i in range(len(corr_pairs)):
        # GET VARIABLES NAMES
        var1 = corr_pairs.iloc[i, 0]
        var2 = corr_pairs.iloc[i, 1]

        # CONCACTENATE VARIABLES WITH TARGET
        df = pd.concat([dataframe[var1], dataframe[var2], target], axis=1)

        # CALCULATE CORRELATION COEFFICIENTS
        corr1 = df.corr(method="kendall").iloc[0, 2]
        corr2 = df.corr(method="kendall").iloc[1, 2]

        # ADD TO LIST THE VARIABLE LESS CORRELATED WITH THE TARGET
        if corr1 < corr2:
            dropping_variables.append(var1)
        else:
            dropping_variables.append(var2)

    # RETURN LIST OF VARIABLES TO DROP WITHOUT DUPLICATES
    return list(set(dropping_variables))


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
