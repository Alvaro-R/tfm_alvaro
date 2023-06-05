"""
Author: Álvaro Román Gómez.

Module to perform model calculations related to training,
preidiction and evaluation of models using GridSearchCV.

"""
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


def build_grid_search_model(model, parameters_grid, cv=5, scoring="roc_auc"):
    """
    Build a grid search model for hyperparameter optimization.

    Args:
        model (BaseEstimator): The model to be optimized.
        parameters_grid (Dict[str, Any]): The grid of hyperparameters to search over.
        cv (int): The number of cross-validation folds (default: 5).
        scoring (Union[str, callable]): The scoring metric used for evaluation
            (default: "roc_auc"). Can be a string or a callable.
    """
    grid_search_model = GridSearchCV(
        model,
        parameters_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
    )
    return grid_search_model


def fit_grid_search_model(grid_search_model, X_train, Y_train):
    """
    Fit a grid search model to the training data.

    Args:
        grid_search_model (GridSearchCV): The grid search model to be fitted.
        X_train (Any): The input features of the training data.
        Y_train (Any): The target values of the training data.

    Returns:
        GridSearchCV: The fitted grid search model.
    """
    grid_search_model.fit(X_train, Y_train)
    return grid_search_model


def get_best_model_from_grid_search(grid_search_model):
    """
    Get the best model from a fitted grid search model.

    Args:
        grid_search_model (GridSearchCV): The fitted grid search model.
    """
    best_model = grid_search_model.best_estimator_
    return best_model


def predict_with_best_model_from_grid_search(best_model, X_test):
    """
    Make predictions using the best model from a grid search.

    Args:
        best_model (BaseEstimator): The best model obtained from grid search.
        X_test (Union[list, np.ndarray]): The input features for prediction.

    Returns:
        Union[list, np.ndarray]: The predicted values.
    """
    Y_pred = best_model.predict(X_test)
    return Y_pred


def predict_and_evaluate_with_best_model_from_grid_search(best_model, X_test, Y_test):
    """
    Make predictions using the best model from a grid search and evaluate the model.

    Args:
        best_model (BaseEstimator): The best model obtained from grid search.
        X_test (Union[list, np.ndarray]): The input features for prediction.
        Y_test (Union[list, np.ndarray]): The true labels for evaluation.

    Returns:
        Union[list, np.ndarray]: The predicted values.
    """
    Y_pred = predict_with_best_model_from_grid_search(best_model, X_test)
    evaluate_model(Y_test, Y_pred)
    return Y_pred


def evaluate_model(Y_test, Y_pred):
    """
    Evaluate a model using various metrics.

    Args:
        Y_test (List[int]): The true labels.
        Y_pred (List[int]): The predicted labels.

    Returns:
        List[float]: A list of evaluation results, including accuracy,
        precision, recall, F1 score, and AUC.
    """
    results = []

    # ACCURACY
    accuracy = metrics.accuracy_score(Y_test, Y_pred)
    results.append(round(accuracy, 2))
    # PRECISION
    precision = metrics.precision_score(Y_test, Y_pred)
    results.append(round(precision, 2))
    # RECALL
    recall = metrics.recall_score(Y_test, Y_pred)
    results.append(round(recall, 2))
    # F1
    f1 = metrics.f1_score(Y_test, Y_pred)
    results.append(round(f1, 2))
    # AUC
    auc = metrics.roc_auc_score(Y_test, Y_pred)
    results.append(round(auc, 2))

    # RETURN EVALUATION RESULTS
    return results


def calculate_models(
    datasets: dict,
    model: object,
    parameters_grid: dict,
    model_name: str,
    cv=5,
    scoring="roc_auc",
):
    """
    Calculate and evaluate models on multiple datasets using grid search.

    Args:
        datasets (Dict[str, Dict[str, Any]]): A dictionary of datasets,
        where the keys are the dataset names and the values
        are dictionaries containing the training and test data.
        model (object): The model object to be used for training.
        parameters_grid (Dict[str, Any]): A dictionary
        specifying the hyperparameter grid for the grid search.
        model_name (str): The name of the model.
        cv (int, optional): The number of cross-validation folds.
        Default is 5.
        scoring (str, optional): The scoring metric to be
        used for grid search.
        Default is "roc_auc".

    Returns:
        Tuple[pd.DataFrame, List[pd.DataFrame]]: A tuple containing a
        dataframe with the evaluation results for each model
        and a list of dataframes containing the ROC curve
        data for each model.
    """
    # CREATE DATAFRAME TO SAVE RESULTS
    models = pd.DataFrame()
    roc_curves = []

    # FOR EACH DATASET, DEFINE THE MODEL
    for dataset_name, dataset in datasets.items():
        # NAME OF MODEL
        name = model_name + "_" + dataset_name

        # INSTANCE GRIDSEARCH MODEL
        grid_search_model = build_grid_search_model(
            model, parameters_grid, cv=cv, scoring=scoring
        )

        # TRAIN MODEL
        grid_search_model.fit(
            dataset[f"X_train_{dataset_name}"], dataset[f"Y_train_{dataset_name}"]
        )

        # AUC FOR BEST MODEL
        auc_training = grid_search_model.best_score_

        # GET BEST MODEL
        best_model = grid_search_model.best_estimator_

        # PREDICT WITH BEST MODEL
        Y_pred = best_model.predict(dataset[f"X_test_{dataset_name}"])

        # EVALUATE MODEL
        evaluation = evaluate_model(dataset[f"Y_test_{dataset_name}"], Y_pred)

        # SAVE RESULTS
        results = pd.DataFrame(
            {
                "model_name": name,
                "accuracy": evaluation[0],
                "precision": evaluation[1],
                "recall": evaluation[2],
                "f1": evaluation[3],
                "auc": evaluation[4],
                "auc_training": auc_training,
                "model_parameters": str(best_model.get_params()),
            },
            index=[0],
        )

        # ROC CURVE FOR BEST MODEL
        fpr, tpr, _ = metrics.roc_curve(dataset[f"Y_test_{dataset_name}"], Y_pred)

        # ROC CURVE DATAFRAME
        roc_curve = pd.DataFrame(
            {
                "model_name": name,
                "fpr": pd.Series(fpr),
                "tpr": pd.Series(tpr),
                "auc": evaluation[4],
            }
        )

        # SAVE ROC CURVE
        roc_curves.append(roc_curve)

        # SAVE RESULTS IN DATAFRAME
        models = pd.concat([models, results], axis=0)

    # RETURN DATAFRAME WITH RESULTS
    return (models, roc_curves)


def plot_roc_curve(roc_curve: pd.DataFrame):
    """
    Plot the ROC curve.

    Args:
        roc_curve (pd.DataFrame): DataFrame containing the ROC curve data.

    Returns:
        None.
    """
    # IMPORT LIBRARIES
    import matplotlib.pyplot as plt

    # PLOT ROC CURVE
    plt.plot(
        roc_curve["fpr"],
        roc_curve["tpr"],
        label=roc_curve["model_name"][0] + "(AUC = %0.2f)" % roc_curve["auc"][0],
    )

    return plt


def plot_all_roc_curves(roc_curves: list, model_name: str, figure_path: str, save=True):
    """
    Plot all ROC curves.

    Args:
        roc_curves (List[pd.DataFrame]): List of DataFrames containing
        the ROC curve data.
        model_name (str): Name of the model.
        figure_path (str): Path to save the figure.
        save (bool, optional): Whether to save the figure. Defaults to True.

    Returns:
        plt: The matplotlib.pyplot object.
    """
    # IMPORT LIBRARIES
    import matplotlib.pyplot as plt

    # FIGURE SIZE
    plt.figure(figsize=(10, 10))

    # PLOT ALL ROC CURVES
    for roc_curve in roc_curves:
        plot_roc_curve(roc_curve)

    # PLOT LEGEND
    plt.plot([0, 1], [0, 1], color="black", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - " + model_name)
    plt.legend(loc="lower right")

    # SAVE FIGURE
    if save:
        plt.savefig(figure_path + model_name + "_roc_curve.png", dpi=300)

    return plt
