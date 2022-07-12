import pandas as pd
import numpy as np

import jinja2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from typing import Tuple

# keras specific imports
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
pd.set_option('mode.chained_assignment', None)


# 1. Data preparation
def convert_ftr(result: str) -> int:
    """Convert the result of ftr to a integer format

    Args:
        result: result of a game, that should be converted

    Returns: converted result

    """
    if result == "D":
        return 1
    elif result == "H":
        return 2
    elif result == "A":
        return 0
    else:
        return None


def print_corrlation_matrix_large(correlation_matrix_df: pd.DataFrame) -> None:
    """prints the correlation matrix for the whole dataframe

        Args:
            correlation_matrix_df (pd.DataFrame): Dataframe for the correlation matrix

    """
    correlation_matrix_match_data_unfiltered = correlation_matrix_df.copy(deep=True).corr()[
        ['FTR', 'FTHG', 'FTAG']]
    correlation_matrix_match_data_unfiltered = correlation_matrix_match_data_unfiltered.dropna()
    correlation_matrix_match_data_unfiltered = correlation_matrix_match_data_unfiltered.iloc[
        6:]
    # correlation_matrix_match_data_unfiltered.style.background_gradient()
    # can only be used with web framework (e.g. jupyter notebook), but much prettier
    plt.imshow(correlation_matrix_match_data_unfiltered, cmap=plt.cm.Blues)
    plt.xticks(range(correlation_matrix_match_data_unfiltered.select_dtypes(['number']).shape[1]),
               correlation_matrix_match_data_unfiltered.select_dtypes(['number']).columns, rotation=45)
    plt.yticks(range(correlation_matrix_match_data_unfiltered.select_dtypes(['number']).shape[0]),
               correlation_matrix_match_data_unfiltered.select_dtypes(['number']).index)
    cb = plt.colorbar()
    cb.ax.tick_params()
    plt.title('Correlation Matrix', fontsize=16)


def print_corrlation_matrix_middle(correlation_matrix_df: pd.DataFrame, print_matrix: bool = False) -> None:
    """prints the correlation matrix for some part of the dataframe

        Args:
            correlation_matrix_df (pd.DataFrame): Dataframe for the correlation matrix
            print_matrix (bool, optional): wether to print the correlation matrix or not

    """
    correlation_matrix_df = correlation_matrix_df.loc[:,
                                                      ["FTR", "FTHG", "FTAG", "HomeEloOld", "AwayEloOld", "DiffEloOld", "HomeAttackOld",
                                                       "HomeDefendOld", "AwayAttackOld", "AwayDefendOld", "DiffDefendOld", "DiffAttackOld",
                                                       "MarketValueDiff", "PDiff3Matches", "PDiff10Matches", 'PDiffAllMatches', 'PQuotAllMatches',
                                                       'DirectComparisonGoalDiff', 'DirectComparisonGoalQuot', 'DirectComparisonHG',
                                                       'DirectComparisonAG']]
    correlation_matrix_match_data_reduced = correlation_matrix_df.corr()[
        ['FTR', 'FTHG', 'FTAG']].iloc[3:]
    if print_matrix:
        plt.imshow(correlation_matrix_match_data_reduced, cmap=plt.cm.Blues)
        plt.xticks(range(correlation_matrix_match_data_reduced.select_dtypes(['number']).shape[1]),
                   correlation_matrix_match_data_reduced.select_dtypes(['number']).columns, rotation=45)
        plt.yticks(range(correlation_matrix_match_data_reduced.select_dtypes(['number']).shape[0]),
                   correlation_matrix_match_data_reduced.select_dtypes(['number']).index)
        cb = plt.colorbar()
        cb.ax.tick_params()
        plt.title('Correlation Matrix', fontsize=16)
    return correlation_matrix_match_data_reduced


def print_corrlation_matrix_small(correlation_matrix_match_data_reduced: pd.DataFrame, features: list[str]) -> None:
    """prints the correlation matrix for the used features of the dataframe

        Args:
            correlation_matrix_df (pd.DataFrame): Dataframe for the correlation matrix
            features: list of used features for the correlation matrix/ training

    """
    correlation_matrix_match_data_reduced = correlation_matrix_match_data_reduced.loc[
        features, :]
    correlation_matrix_match_data_reduced.style.background_gradient()
    plt.imshow(correlation_matrix_match_data_reduced, cmap=plt.cm.Blues)
    plt.xticks(range(correlation_matrix_match_data_reduced.select_dtypes(['number']).shape[1]),
               correlation_matrix_match_data_reduced.select_dtypes(['number']).columns, rotation=45)
    plt.yticks(range(correlation_matrix_match_data_reduced.select_dtypes(['number']).shape[0]),
               correlation_matrix_match_data_reduced.select_dtypes(['number']).index)
    cb = plt.colorbar()
    cb.ax.tick_params()
    plt.title('Correlation Matrix', fontsize=16)


def split_train_test_data(features: list, match_data_bl: pd.DataFrame) -> Tuple[np.ndarray]:
    """split data random in train and test data

    Args:
        features: list of features used for the model-training
        match_data_bl: DataFrame containing all match data

    Returns:
        X: np.ndarray with all features and all games
        y: np.ndarray with all game results
        X_train: np.ndarray with random train data
        X_test: np.ndarray with random test data
        y_train: np.ndarray with random train data results
        y_test: np.ndarray with random test data results

    """
    y = match_data_bl.loc[:, ["FTR"]]
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        match_data_bl, y, test_size=0.33, random_state=42)
    X_train = X_train_df.loc[:, features]
    X_test = X_test_df.loc[:, features]
    X = match_data_bl.loc[:, features]
    return X, y, X_train, X_test, y_train, y_test


def one_hot_encode_outputs(y_train, y_test) -> Tuple[np.ndarray, np.ndarray]:
    """one hot encode outputs of result to also predict probabilities later

    Args:
        y_train: np.ndarray with random train data results
        y_test: np.ndarray with random test data results

    Returns:
        modified np.ndarray with one hot encoded outputs

    """
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    count_classes = y_test.shape[1]
    print(count_classes)
    return y_train, y_test


def split_train_test_data_for_predicting_s_22(X: np.ndarray, y: np.ndarray, match_data_bl: pd.DataFrame) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """split train test data to predict one season (season 2022)

    Args:
        X: np.ndarray with all features and all games
        y: np.ndarray with all game results
        match_data_bl: DataFrame containing all match data

    Returns:
        X_train_s_2022: np.ndarray with all train-features and all games except last season
        X_test_s_2022: np.ndarray with all train-features and all games from last season
        y_train_s_2022: np.ndarray with all game results except last season
        y_test_s_2022: np.ndarray with all game results from last season
        match_data_bl_wo_nan_s_2022: modified df without last season data

    """
    # take the last 306 - 10*9 datapoints (because first 10 Matches are needed for the form)
    X_train_s_2022, X_test_s_2022 = X.iloc[:-216], X.iloc[-216:]
    y_train_s_2022, y_test_s_2022 = y[:-216], y[-216:]
    match_data_bl_wo_nan_s_2022 = match_data_bl.iloc[-216:]
    y_train_s_2022, y_test_s_2022 = one_hot_encode_outputs(
        y_train_s_2022, y_test_s_2022)
    return X_train_s_2022, X_test_s_2022, y_train_s_2022, y_test_s_2022, match_data_bl_wo_nan_s_2022


# 2. train model
def train_final_model(X_train, y_train, number_of_epochs: int = 20, validation_data=None) -> Tuple:
    """Train the final model on the training set.

    Args:
        X_train: np.ndarray with all train data features
        y_train:  np.ndarray with all results for the training set
        number_of_epochs: int number of epochs to train the model on
        validation_data: (optional) Tuple with X_test and y_test data; will use random split of X_train and y_train if
                         nothing is specified

    Returns:
        Tuple of trained_model and model history

    """
    # create model
    model = Sequential()

    # get number of columns in training data
    n_cols_2 = X_train.shape[1]
    # y_train = to_categorical(y_train)
    # add layers to model
    model.add(Dense(80, activation='sigmoid', input_dim=n_cols_2))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(70, activation='sigmoid'))
    model.add(Dense(3, activation='softmax'))

    # compile model using accuracy to measure model performance
    model.compile(metrics=['accuracy'], optimizer='adam',
                  loss='binary_crossentropy')

    if validation_data is None:
        history = model.fit(
            X_train, y_train, epochs=number_of_epochs, validation_split=0.3)
    else:
        history = model.fit(
            X_train, y_train, epochs=number_of_epochs, validation_data=validation_data)
    return model, history


# 3. evaluate model
def print_accuracy(X_train, y_train, X_test, y_test, model) -> Tuple[np.ndarray, np.ndarray]:
    """prints the accuracy of the model on test and train data

    Args:
        X_train: np.ndarray with train data
        y_train: np.ndarray with random train data results
        X_test: np.ndarray with test data
        y_test: np.ndarray with random test data results
        model: model that should be evaluated

    Returns:
        pred_train: np.ndarray with model predictions on training data
        pred_test: np.ndarray with model predictions on test data

    """
    pred_train = model.predict(X_train)
    if y_train.shape != pred_train.shape:
        y_train = to_categorical(y_train)
    scores = model.evaluate(X_train, y_train, verbose=0)
    print('Accuracy on training data: {}% \n Error on training data: {}'.format(
        scores[1], 1 - scores[1]))

    pred_test = model.predict(X_test)
    if y_test.shape != pred_test.shape:
        y_test = to_categorical(y_test)
    scores2 = model.evaluate(X_test, y_test, verbose=0)
    print('Accuracy on test data: {}% \n Error on test data: {}'.format(
        scores2[1], 1 - scores2[1]))
    return pred_train, pred_test


def get_predictions_from_model(pred_test: np.ndarray, alpha: float = 0.0) -> list:
    """reconverts the outputs of the model to one dimension based on the probabilities for each class (home, draw, away)
    If the probabilities for home and away are almost equal (compared to value alpha), then it will also predict draw

    Args:
        pred_test: np.ndarray with model predictions on test data
        alpha: Threshold value how much the probabilities for home and away have to differ

    Returns: one-dimensional list with predicted results

    """
    y_pred = []
    for test in pred_test:
        if test[2] > test[1] and test[2] > test[0] and abs(test[2] - test[0]) > alpha:
            y_pred.append(2)
        elif test[0] > test[1] and abs(test[2] - test[0]) > alpha:
            y_pred.append(0)
        else:
            y_pred.append(1)
    return y_pred


def get_predictions_from_test(y_test: np.ndarray) -> list:
    """reconverts the test array into a one dimensional list

    Args:
        y_test: np.ndarray with test data

    Returns:  one-dimensional list with test data

    """
    y_test_res = []
    for test in y_test:
        for counter, j in enumerate(test):
            if np.max(test) == j:
                y_test_res.append(counter)
    return y_test_res


def plot_train_test_accuracy(history) -> None:
    """summarize and plot history for accuracy

    Args:
        history: history of the trained model

    Returns: None

    """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def plot_loss_history(history) -> None:
    """summarize and plots history for loss

    Args:
        history: history of the trained model

    Returns: None

    """
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def predict_season_2022(X_test_s_2022: np.ndarray, model: Sequential, match_data_bl_wo_nan_s_2022: pd.DataFrame) \
        -> np.ndarray:
    """predicts last season and saves results to csv

    Args:
        X_test_s_2022: np.ndarray with all train-features and all games from last season
        model: trained model to make predictions on
        match_data_bl_wo_nan_s_2022: Data array with all data from last season

    Returns: np.ndarray with predictions for last season

    """
    data = model.predict(X_test_s_2022)
    predicted_data_season_2022 = get_predictions_from_model(data, alpha=0.05)
    predicted_data_season_prob = pd.DataFrame(data, columns=['ProbAwayWin', 'ProbDraw', 'ProbHomeWin'],
                                              index=match_data_bl_wo_nan_s_2022.index)
    match_data_bl_wo_nan_s_2022.loc[:, [
        'ProbAwayWin', 'ProbDraw', 'ProbHomeWin']] = predicted_data_season_prob
    match_data_bl_wo_nan_s_2022.loc[:,
                                    'predictedResults'] = predicted_data_season_2022
    match_data_bl_wo_nan_s_2022.to_csv("Data\\match_data_predicted_bl_22.csv")
    return predicted_data_season_2022


def print_confusion_matrix(y_test_res: list, y_pred: list) -> None:
    """prints the confusion matrix for actual test and predicted results

    Args:
        y_test_res: list of test results
        y_pred: list of predicted results

    Returns: None

    """
    cf = confusion_matrix(y_test_res, y_pred)
    cfd = ConfusionMatrixDisplay(
        cf, display_labels=["Away", "Draw", "Home"])
    cfd.plot(cmap=plt.cm.Blues)
    plt.show()


if __name__ == '__main__':
    # read the csv file
    match_data_unfiltered = pd.read_csv(
        r"Data\preprocessed_dataframe_with_elo_mw_form_final_version.csv")
    # convert full time result to integer
    match_data_unfiltered['FTR'] = match_data_unfiltered['FTR'].apply(
        convert_ftr)
    # drop unnecessary lines
    match_data_bl = match_data_unfiltered.drop(
        ["Unnamed: 0"], axis=1, errors="ignore")

    # print correlation matrix
    reduced_match_data_bl_corr = print_corrlation_matrix_middle(
        match_data_bl, print_matrix=False)
    features = ["DiffEloOld", "DiffAttackOld", "DiffDefendOld", "PDiff3Matches", "PDiff10Matches", "PDiffAllMatches",
                "MarketValueDiff", "DirectComparisonHG", "DirectComparisonAG"]
    print_corrlation_matrix_small(
        reduced_match_data_bl_corr, features=features)

    # predict for random data
    match_data_bl_wo_nan = match_data_bl.dropna(subset=features)
    X, y, X_train, X_test, y_train, y_test = split_train_test_data(
        features, match_data_bl_wo_nan)
    model_random, history_season_random = train_final_model(
        X, to_categorical(y), number_of_epochs=20)
    predictions = model_random.predict(X_test)
    predictions = get_predictions_from_model(predictions, alpha=0.15)
    print_confusion_matrix(y_test, predictions)
    print_accuracy(X_train, y_train, X_test, y_test, model_random)
    plot_train_test_accuracy(history_season_random)
    plot_loss_history(history_season_random)

    # predict for season 2022
    X_train_s_2022, X_test_s_2022, y_train_s_2022, y_test_s_2022, match_data_bl_wo_nan_s_2022 \
        = split_train_test_data_for_predicting_s_22(X, y, match_data_bl_wo_nan)
    model_season_2022, history_season_2022 = train_final_model(X_train_s_2022, y_train_s_2022, number_of_epochs=20,
                                                               validation_data=(X_test_s_2022, y_test_s_2022))
    y_prediction_season_2022 = predict_season_2022(
        X_test_s_2022, model_season_2022, match_data_bl_wo_nan_s_2022)
    print_accuracy(X_train_s_2022, y_train_s_2022,
                   X_test_s_2022, y_test_s_2022, model_season_2022)
    # print confusion matrix
    print_confusion_matrix(get_predictions_from_test(
        y_test_s_2022), y_prediction_season_2022)
