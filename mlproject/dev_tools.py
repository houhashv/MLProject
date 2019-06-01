import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import numpy as np


def best_t_f1(precisions, recalls, thresholds):
    """
    calculate the best threshold by F1 measure
    :param precisions: precisions from the precision-recall curve - list of float
    :param recalls: recalls from the precision - recall curve - list of float
    :param thresholds: thresholds from the precision-recall curve - list of float
    :return: the best threshold - float
    """
    f1 = [2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i]) for i in range(0, len(thresholds))]
    t = thresholds[np.argmax(f1)]
    return t, max(f1)


def is_date(df, col, string_ratio=0.02):
    """
    check if a column in a dataframe is a date or not
    :param df: the dataframe to check if it's ok - Dataframe
    :param col: the column to operate over - string
    :param string_ratio: the ratio that deside if there failed attempts / size of vector percentage larger than ratio
    than the feature is not a date - float
    :return: if the column is a date or not - boolearn
    """
    count = 1
    value_list = df[col].fillna(0).unique().tolist()

    for value in value_list:
        try:
            pd.Timestamp(value)
        except Exception as e:
            count += 1
            if count / len(value_list) >= string_ratio:
                return False

    return True


def get_cols(df, exclude=[], ratio=0.01):
    """
    finds the diffrent types of features automatically
    :param df: the dataframe to check for columns - Dataframe
    :param exclude: columns to ignore - list of strings
    :param ratio: this ratio decide if a numeric column is actually categoric if the number of unique values in the
    feature devided by the size of feature is less than the ratio - float
    :return: dictionary with keys as columns types and values that are lists of strings with column names - dictionary
    """
    key_cols = [col for col in df.columns.tolist() if "_ID" in col]
    cat_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()
    date_cols = [col for col in cat_cols if col not in exclude + key_cols and is_date(df, col)]
    cat_cols = [col for col in cat_cols if col not in date_cols + key_cols + exclude]
    numeric_cols = [col for col in df.columns.tolist() if col not in key_cols + date_cols + cat_cols + exclude]

    for col in numeric_cols:
        if df[col].unique().shape[0] <= ratio * df.shape[0]:
            cat_cols.append(col)

    numeric_cols = [col for col in numeric_cols if col not in cat_cols]

    return {"key": key_cols, "categoric": cat_cols, "date": date_cols, "numeric": numeric_cols}


def show_descriptive(df, col):
    """
    show the distribution of in histogram and over a one dimetional graph
    :param df: the dataframe with the data - Dataframe
    :param col: the column to show data of - string
    """
    print(col)
    se = df[col].describe()
    print(se)
    print(se.shape[0])
    try:
        df[col].hist()
        plt.show()
    except:
        print("can't show histogram, there is a problem with the values")
    plt.scatter(df[col], [0 for x in range(0, df.shape[0])])
    plt.show()


def datetime_extract(df, col):
    """
    extract the date and time to new columns from a datetime column
    :param df: the dataframe to perform the operation over - Dataframe
    :param col: the column to chcek - string
    :return: a dataframe of the result - Dataframe
    """
    result_date = df[col].str.extractall(r'(?P<' + col + '_date>(?P<' + col + '_year>\d\d\d\d)-(?P<' + col + '_month>\d\d)-(?P<' + col + '_day>\d?\d) (?P<' + col + '_time>(?P<' + col + '_hours>\d?\d):(?P<' + col + '_minutes>\d\d):(?P<' + col + '_seconds>\d\d)))').reset_index()
    if result_date.shape[0] > 0:
        try:
            result = result_date.drop(["level_0", "match", col + "_date", col + "_time"], axis=1)
        except Exception as e:
            print(e)
    return result


def sampler(df, rows_per_min=0.2, rows_per_max=0.8, cols_per_min=0.2, cols_per_max=0.8, n=10, repeat=True, target=None):
    """
    sample n dataframes from one dataframe for each percentage limit over rows and cols
    :param df: the data frame
    :param rows_per_min: lower percentage of rows to sample, default: 0.2 - float
    :param rows_per_max: higher percentage of rows to sample, default: 0.8 - float
    :param cols_per_min: lower percentage of columns to sample, default: 0.2 - float
    :param cols_per_max: higher percentage of columns to sample, default: 0.8 - float
    :param n: the number of dataframes to sample, default: 10 - int
    :param repeat: can we repeat the same row or column indicator, default: True - boolean
    :param target: the target column to ignore - string
    :return: list of all the sampled dataframes - list of Dataframe
    """
    try:
        df_target = df.copy(True).drop([target], axis=1)
    except Exception as e:
        print("the dataframe is None")
        raise e

    cols = df_target.columns
    rows = df_target.index
    dfs = []
    exclude_index = []
    exclude_columns = []

    for row_measure in [rows_per_min, rows_per_max]:
        for col_measure in [cols_per_min, cols_per_max]:
            for i in range(0, n):
                if repeat:
                    remained_index = rows[random.sample(range(len(rows) - 1), int(row_measure * len(rows)))]
                    remained_cols = cols[random.sample(range(len(cols) - 1), int(col_measure * len(cols)))]
                else:
                    remained_index = pd.Index([row for row in rows if row not in exclude_index])
                    remained_index = remained_index[random.sample(range(len(remained_index) - 1),
                                                                  int(row_measure * len(remained_index)))]
                    remained_cols = pd.Index([col for col in cols if col not in exclude_columns])
                    remained_cols = remained_cols[random.sample(range(len(remained_cols) - 1),
                                                                int(col_measure * len(remained_cols)))]
                df_i = df_target.iloc[remained_index, :]
                exclude_index += list(df_i.index)
                df_i_c = df_i[remained_cols]
                exclude_columns += list(df_i_c.columns)
                df_i_c[target] = df[target]
                if len(df_i_c.columns) > len(set(df_i_c.columns)):
                    print(1)
                if len(df_i_c.index) > len(set(df_i_c.index)):
                    print(1)
                dfs.append(df_i_c)

            exclude_columns = []
            exclude_index = []

    return dfs


class Logger:
    """
    used to log important data
    """
    def __init__(self):
        """
        constructor
        """
        pass

    @staticmethod
    def to_log(error_msg, log_file):
        """
        log a message
        :param error_msg: the error mesage to log - string
        :param log_file: the file name to log to - string
        """
        try:
            with open(os.getcwd() + '\\logs\\{}.log'.format(log_file), 'a') as i:
                i.write(error_msg)
        except Exception as e:
            print(e)
