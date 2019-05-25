import mlproject.datasets.datasets.xgboost_data as xgboostdata
from mlproject.dev_tools import get_cols
from mlproject.pre_processing.pipelines import MLPipeline
from mlproject.pre_processing.transformers import *
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
import copy
import shap
import pickle
import os


def run():

    key_cols = ['id', 'siduri', 'misparprat', 'wave']
    target_cols = ["futereLFP", "future_hour_income", "incomeGroupUp"]
    exclude = ["futureIncomeGroup", "future_mean_income_group", "incomeGroupDown", 'misparmeshivproxy', 'mishkalorech',
               'nGroup', 'n', 'future_hour_income', 's_seker', 'gilGroup', 'paamachronachashavani', 'gilm',
               'Unnamed: 0']

    df_p = xgboostdata.dflearning()
    df_meshek = xgboostdata.dflearning_mb()
    dflearning = pd.merge(df_p, df_meshek, on='siduri', how='left')
    score_p = xgboostdata.dfscoring()
    score_m = xgboostdata.dfscoring_mb()
    dfscore = pd.merge(score_p, score_m, on='siduri', how='left')

    columns_prep = get_cols(dflearning, key_cols + target_cols + exclude, 0.005)
    problems = [str(i) for i in range(1, 4)]


    columns = dict()
    for index, p in enumerate(problems):
        columns[p] = None
        columns[p] = copy.deepcopy(columns_prep)
        columns[p]["key"] = key_cols
        columns[p]["target"] = [target_cols[index]]
    # dfs = sampler(dflearning)
    problem = "3"
    cat_cols = columns[problem]["categoric"]
    numeric_cols = columns[problem]["numeric"]
    target = columns[problem]["target"]
    dflearning = dflearning[cat_cols + numeric_cols + target]
    X_train, X_test, y_train, y_test = train_test_split(dflearning.drop(target, axis=1), dflearning[target],
                                                        test_size=0.3, random_state=42)
    complete_cols = [col for col in cat_cols + numeric_cols if col not in dfscore.columns]
    for col in complete_cols:
        dfscore[col] = 0
    # X_out_of_time = dfscore[cat_cols + numeric_cols]

    clear_stage = ClearNoCategoriesTransformer(categorical_cols=cat_cols)
    imputer = ImputeTransformer(numerical_cols=numeric_cols, categorical_cols=cat_cols)
    outliers = OutliersTransformer(numerical_cols=numeric_cols, categorical_cols=cat_cols)
    scale = ScalingTransformer(numerical_cols=numeric_cols)
    categorize = CategorizeByTargetTransformer(categorical_cols=cat_cols)
    correlations = CorrelationTransformer(numerical_cols=numeric_cols, categorical_cols=cat_cols, target=target,
                                          threshold=0.9)
    # le = LabelEncoderTransformer(cat_cols)
    dummies = DummiesTransformer(cat_cols)

    clf1 = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    clf2 = RandomForestClassifier(random_state=0, n_jobs=-1, max_depth=1)
    clf3 = LogisticRegression()
    clf4 = MLPClassifier()
    clf5 = GaussianNB()
    meta_cls = XGBClassifier(n_jobs=-1, alpha=10, max_depth=3, objective="reg:logistic")
    meta = StackingClassifier(classifiers=[clf3,clf4], meta_classifier=meta_cls)
    # Number of trees in random forest
    n_estimators = [200]
    max_features = ['auto']
    max_depth = [20]
    min_samples_split = [5]
    min_samples_leaf = [2]
    bootstrap = [True]
    # Create the random grid
    random_grid = {'classifier__randomforestclassifier__n_estimators': n_estimators,
                   'classifier__randomforestclassifier__max_features': max_features,
                   'classifier__randomforestclassifier__max_depth': max_depth,
                   'classifier__randomforestclassifier__min_samples_split': min_samples_split,
                   'classifier__randomforestclassifier__min_samples_leaf': min_samples_leaf,
                   'classifier__randomforestclassifier__bootstrap': bootstrap,
                   'classifier__meta-xgbclassifier__n_estimators': n_estimators,
                   'classifier__meta-xgbclassifier__max_features': max_features,
                   'classifier__meta-xgbclassifier__max_depth': max_depth,
                   'classifier__meta-xgbclassifier__min_samples_split': min_samples_split,
                   'classifier__meta-xgbclassifier__min_samples_leaf': min_samples_leaf,
                   'classifier__meta-xgbclassifier__bootstrap': bootstrap}

    params = random_grid
    steps_feat = [("clear_non_variance", clear_stage),
                  ("imputer", imputer),
                  ("outliers", outliers),
                  ("scaling", scale),
                  ("categorize", categorize),
                  ("correlations", correlations),
                  ("dummies", dummies)]
    pipeline_feat = MLPipeline(steps=steps_feat)
    X_train = pipeline_feat.fit_transform(X_train, y_train).reset_index(drop=True)
    X_test = pipeline_feat.transform(X_test).reset_index(drop=True)
    path = os.getcwd() + "/results/"
    X_train.to_csv(path + "x_train.csv", index=False)
    X_test.to_csv(path + "x_test.csv", index=False)
    y_train.to_csv(path + "y_train.csv", index=False)
    y_test.to_csv(path + "x_test.csv", index=False)
    steps = [("classifier", meta)]
    pipeline = MLPipeline(steps=steps)
    grid = GridSearchCV(estimator=pipeline, param_grid=params, cv=2, refit=True)
    grid.fit(X_train, y_train)
    train_score = grid.score(X_train, y_train)
    test_score = grid.score(X_test, y_test)
    pickle.dump(grid, open(path + "grid.p", "wb"))
    # grid = pickle.load(open("/{}.p".format(problem), "rb"))
    print("problem {} results".format(problem))
    print("train_score: {}".format(train_score))
    print("test_score: {}".format(test_score))


def analysis():

    path = os.getcwd()
    X_train = pd.read_csv(path + "/results/x_train.csv")
    y_train = pd.read_csv(path + "/results/x_train.csv")
    X_test = pd.read_csv(path + "/results/x_train.csv")
    grid = pickle.load(open(path + "/results/grid.p", "rb"))
    estimator = grid.predict_proba(X_test)[:, 1]
    f = lambda x: grid.best_estimator_.steps[-1][1].predict_proba(x)[:, 1]
    X = X_train
    X = pd.concat([X, y_train], axis=1)
    a = X.corr("spearman")
    print(a)
    # use Kernel SHAP to explain test set predictions
    explainer = shap.KernelExplainer(f, X)
    shap_values = explainer.shap_values(X[0:10])
    print(estimator)
    print(sum(estimator) / estimator.shape[0])
    shapv = pd.DataFrame(shap_values)
    base_pred = explainer.expected_value
    explainer.expected_value
    shapv.index = X[0:10].index
    shapv.columns = 's_' + X.columns
    shap_all = pd.concat([shapv, X[0:10]], axis=1)
    a = shap_all.query('gil<50').sort_values(['gil']).index.sort_values()
    b = shapv.loc[a,:].sum(axis=1).sort_values(ascending=True).index
    shapv.loc[b, :]
    shap.force_plot(explainer.expected_value, shap_values[8, :], X.iloc[8, :], matplotlib=True)
    shap.summary_plot(shap_values, X[0:10], max_display=40, plot_type="dot")
    plt.show()


if __name__=="__main__":

    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    run()
    # analysis()
