import pandas as pd
import numpy as np

import jinja2
from sklearn.preprocessing import scale as scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from typing import Tuple

# keras specific imports
from tensorflow import keras
from keras_tuner.tuners import RandomSearch
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical


# 1. Data preparation

def convert_ftr(result: str):
    """ make FTR column to continous variable


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
    correlation_matrix_match_data_unfiltered.style.background_gradient()


def print_corrlation_matrix_middle(correlation_matrix_df: pd.DataFrame) -> None:
    """prints the correlation matrix for some part of the dataframe

        Args:
            correlation_matrix_df (pd.DataFrame): Dataframe for the correlation matrix

    """
    correlation_matrix_df = correlation_matrix_df.loc[:,
                                                      ["FTR", "FTHG", "FTAG", "HomeEloOld", "AwayEloOld", "DiffEloOld", "HomeAttackOld",
                                                       "HomeDefendOld", "AwayAttackOld", "AwayDefendOld", "DiffDefendOld", "DiffAttackOld",
                                                       "MarketValueDiff", "PDiff3Matches", "PDiff10Matches", 'PDiffAllMatches', 'PQuotAllMatches',
                                                       'DirectComparisonGoalDiff', 'DirectComparisonGoalQuot', 'DirectComparisonHG',
                                                       'DirectComparisonAG']]
    correlation_matrix_match_data_reduced = correlation_matrix_df.corr()[
        ['FTR', 'FTHG', 'FTAG']].iloc[3:]
    correlation_matrix_match_data_reduced.style.background_gradient()
    return correlation_matrix_match_data_reduced


def print_corrlation_matrix_small(correlation_matrix_match_data_reduced: pd.DataFrame, features: list[str]) -> None:
    """prints the correlation matrix for the used features of the dataframe

        Args:
            correlation_matrix_df (pd.DataFrame): Dataframe for the correlation matrix
            features: list of used features

    """
    correlation_matrix_match_data_reduced = correlation_matrix_match_data_reduced.loc[
        features, :]
    correlation_matrix_match_data_reduced.style.background_gradient()


def split_train_test_data(features: list, match_data_bl: pd.DataFrame) -> Tuple[np.ndarray]:
    """

    Args:
        features:
        match_data_bl:

    Returns:

    """
    y = match_data_bl_wo_nan.loc[:, ["FTR"]]
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        match_data_bl_wo_nan, y, test_size=0.33, random_state=42)
    X_train = X_train_df.loc[:, features]
    X_test = X_test_df.loc[:, features]
    X = match_data_bl_wo_nan.loc[:, features]
    return X, y, X_train, X_test, y_train, y_test


def one_hot_encode_outputs(y_train, y_test) -> Tuple[np.ndarray, np.ndarray]:
    # one hot encode outputs
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    count_classes = y_test.shape[1]
    print(count_classes)
    return y_train, y_test

# def train_model(X_train, y_train, number_of_epochs=20):
#     model = Sequential()
#     # model.add(Dense(500, activation='relu', input_dim=6))
#     # model.add(Dense(100, activation='relu'))
#     # model.add(Dense(50, activation='relu'))
#     # model.add(Dense(3, activation='softmax'))
#
#     model.add(Dense(500, activation='relu', input_dim=X_train.shape[1]))
#     model.add(Dense(100, activation='relu'))
#     model.add(Dense(50, activation='relu'))
#     model.add(Dense(3, activation='softmax'))
#
#     # Compile the model
#     model.compile(optimizer='adam',
#                 loss='categorical_crossentropy',
#                 metrics=['accuracy'])
#     # build the model
#     history = model.fit(X_train, y_train, epochs=number_of_epochs)
#     return history, model


def print_accuracy(X_train, y_train, X_test, y_test, model):
    pred_train = model.predict(X_train)
    scores = model.evaluate(X_train, y_train, verbose=0)
    print('Accuracy on training data: {}% \n Error on training data: {}'.format(
        scores[1], 1 - scores[1]))

    pred_test = model.predict(X_test)
    scores2 = model.evaluate(X_test, y_test, verbose=0)
    print('Accuracy on test data: {}% \n Error on test data: {}'.format(
        scores2[1], 1 - scores2[1]))
    return pred_train, pred_test


def get_predictions_from_model(pred_test, alpha: float = 0.):
    y_pred = []
    for test in pred_test:
        if test[2] > test[1] and test[2] > test[0] and abs(test[2] - test[0]) > alpha:
            y_pred.append(2)
        elif test[0] > test[1] and abs(test[2] - test[0]) > alpha:
            y_pred.append(0)
        else:
            y_pred.append(1)
    return y_pred


def get_predictions_from_test(y_test: list):
    y_test_res = []
    for test in y_test:
        for counter, j in enumerate(test):
            if np.max(test) == j:
                y_test_res.append(counter)
    return y_test_res


def print_confusion_matrix(y_test_res, y_pred):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    # , labels=["Draw", "Home", "Away"])
    cf = confusion_matrix(y_test_res, y_pred)
    cfd = ConfusionMatrixDisplay(cf, display_labels=["Away", "Draw", "Home"])
    cfd.plot(cmap=plt.cm.Blues)

# ## Train second NN

# def build_model(hp):
#     model = keras.Sequential()
#     # model.add(Dense(keras.layers.Flatten(input_dim=n_cols_2)))
#     model.add(keras.layers.Flatten())
#     # model.add(layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=128), activation=hp.Choice("activation", ["relu", "tanh"]), input_dim=n_cols_2))
#     for i in range(hp.Int('layers', 2, 6)):
#         model.add(keras.layers.Dense(
#             units=hp.Int('units_' + str(i), 16, 256, step=32),
#             activation=hp.Choice('act_' + str(i), ['relu', 'sigmoid'])))
#     model.add(keras.layers.Dense(3, activation='softmax'))
#     model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
#     return model
#
# def run_random_search(X_train, y_train, X_test, y_test):
#
#     tuner = RandomSearch(
#         build_model,
#         objective='val_accuracy',
#         max_trials=50,
#         executions_per_trial=4,
#         directory='my_dir2',
#         project_name='data_exploration'
#     )
#
#     tuner.search_space_summary()
#     tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
#     tuner.results_summary()


def split_train_test_data_for_predicting_s_22(X, y, match_data_bl_wo_nan):
    # split train test data to predict one season
    # take the last 306 - 10*9 datapoints (because first 10 Matches are needed for the form)

    X_train_s_2022, X_test_s_2022 = X.iloc[:-216], X.iloc[-216:]
    y_train_s_2022, y_test_s_2022 = y[:-216], y[-216:]
    # ToDo delete later
    match_data_bl_wo_nan_s_2022 = match_data_bl_wo_nan.iloc[-216:]
    y_train_s_2022, y_test_s_2022 = one_hot_encode_outputs(
        y_train_s_2022, y_test_s_2022)
    return X_train_s_2022, X_test_s_2022, y_train_s_2022, y_test_s_2022, match_data_bl_wo_nan_s_2022


def train_final_model(X_train, y_train, number_of_epochs: int = 20, validation_data=None):
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
        # , callbacks=[early_stopping_monitor]) #callbacks=[es],
        history = model.fit(
            X_train, y_train, epochs=number_of_epochs, validation_data=validation_data)
    return model, history


def plot_train_test_accuracy(history) -> None:
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def plot_loss_history(history) -> None:
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def predict_season_2022(X_test_s_2022, model, match_data_bl_wo_nan_s_2022) -> None:
    data = model.predict(X_test_s_2022)
    predicted_data_random_results = get_predictions_from_model(data)
    predicted_data_random = pd.DataFrame(data, columns=[
                                         'ProbAwayWin', 'ProbDraw', 'ProbHomeWin'], index=match_data_bl_wo_nan_s_2022.index)
    match_data_bl_wo_nan_s_2022.loc[:, [
        'ProbAwayWin', 'ProbDraw', 'ProbHomeWin']] = predicted_data_random
    match_data_bl_wo_nan_s_2022.loc[:,
                                    'predictedResults'] = predicted_data_random_results
    match_data_bl_wo_nan_s_2022.to_csv(".//match_data_predicted_bl_22.csv")
    return predicted_data_random_results


def print_confusion_matrix(y_test_res, y_pred):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    # , labels=["Draw", "Home", "Away"])
    cf = confusion_matrix(y_test_res, y_pred)
    cfd = ConfusionMatrixDisplay(cf, display_labels=["Away", "Draw", "Home"])
    cfd.plot(cmap=plt.cm.Blues)
    plt.show()


if __name__ == '__main__':
    # read the csv file
    match_data_unfiltered = pd.read_csv(
        r"preprocessed_dataframe_with_elo_mw_form_final_version.csv")
    match_data_unfiltered['FTR'] = match_data_unfiltered['FTR'].apply(
        convert_ftr)
    # drop unnecessary lines
    match_data_unfiltered = match_data_unfiltered.drop(
        ["Unnamed: 0"], axis=1, errors="ignore")
    match_data_unfiltered.tail(n=5)

    match_data_bl = match_data_unfiltered.copy(deep=True)

    # print correlation matrix
    print_corrlation_matrix_large(match_data_bl)
    reduced_match_data_bl_corr = print_corrlation_matrix_middle(match_data_bl)
    features = ["DiffEloOld", "DiffAttackOld", "DiffDefendOld", "PDiff3Matches", "PDiff10Matches", "PDiffAllMatches",
                "MarketValueDiff", "DirectComparisonHG", "DirectComparisonAG"]
    print_corrlation_matrix_small(
        reduced_match_data_bl_corr, features=features)

    # predict for random data
    match_data_bl_wo_nan = match_data_bl.dropna(subset=features)
    X, y, X_train, X_test, y_train, y_test = split_train_test_data(
        features, match_data_bl_wo_nan)
    model_random, history_season_random = train_final_model(
        X, to_categorical(y), number_of_epochs=15)
    predictions = model_random.predict(X_test)
    print_confusion_matrix(
        y_test, get_predictions_from_model(predictions, alpha=0.1))

    # predict for season 2022
    X_train_s_2022, X_test_s_2022, y_train_s_2022, y_test_s_2022, match_data_bl_wo_nan_s_2022\
        = split_train_test_data_for_predicting_s_22(X, y, match_data_bl_wo_nan)

    model_season_2022, history_season_2022 = train_final_model(X_train_s_2022, y_train_s_2022, number_of_epochs=15,
                                                               validation_data=(X_test_s_2022, y_test_s_2022))

    predict_season_2022(X_test_s_2022, model_season_2022,
                        match_data_bl_wo_nan_s_2022)

    y_prediction_season_2022 = predict_season_2022(
        X_test_s_2022, model_season_2022, match_data_bl_wo_nan_s_2022)

    print_confusion_matrix(get_predictions_from_test(
        y_test_s_2022), y_prediction_season_2022)
