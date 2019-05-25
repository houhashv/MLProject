import pandas as pd
import os
import numpy as np


def dflearning():

    df = pd.read_csv(os.path.dirname(__file__) + "\dflearning.csv", low_memory=False).replace(np.Inf, np.nan)\
        .query('incomeGroup<10')
    try:
        df = df.drop("Unnamed: 0", axis=1)
        return df
    except KeyError:
        pass
    return df


def dflearning_mb():

    df = pd.read_csv(os.path.dirname(__file__) + "\dflearningMB.csv", low_memory=False).replace(np.Inf, np.nan)
    try:
        df = df.drop("Unnamed: 0", axis=1)
        return df
    except KeyError:
        pass
    return df


def dfscoring():

    df = pd.read_csv(os.path.dirname(__file__) + "\dfscoring.csv", low_memory=False).replace(np.Inf, np.nan)\
        .query('incomeGroup<10')
    try:
        df = df.drop("Unnamed: 0", axis=1)
        return df
    except KeyError:
        pass
    return df


def dfscoring_mb():

    df = pd.read_csv(os.path.dirname(__file__) + "\dfscoringMB.csv", low_memory=False).replace(np.Inf, np.nan)
    try:
        df = df.drop("Unnamed: 0", axis=1)
        return df
    except KeyError:
        pass
    return df
