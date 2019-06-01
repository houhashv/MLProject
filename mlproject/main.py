import mlproject.datasets.datasets.xgboost_data as xgboostdata
from mlproject.dev_tools import get_cols, measures
from mlproject.pre_processing.pipelines import MLPipeline
from mlproject.pre_processing.transformers import *
import pandas as pd
import copy
import shap
import pickle
import os
import time
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import multiprocessing as mp


def run():

    key_cols = ['id', 'siduri', 'misparprat', 'wave']
    target_cols = ["futereLFP", "future_hour_income", "incomeGroupUp"]
    exclude = ["futureIncomeGroup", "future_mean_income_group", "incomeGroupDown", 'misparmeshivproxy',
               'mishkalorech', 'nGroup', 'n', 'future_hour_income', 's_seker', 'gilGroup', 'paamachronachashavani',
               'gilm', 'Unnamed: 0']

    df_p = xgboostdata.dflearning()
    df_meshek = xgboostdata.dflearning_mb()
    dflearning = pd.merge(df_p, df_meshek, on='siduri', how='left')
    dflearning = dflearning[dflearning["incomeGroup"] < 10]
    columns_prep = get_cols(dflearning, key_cols + target_cols + exclude, 0.005)
    problems = [str(i) for i in range(1, 4)]
    columns = dict()
    for index, p in enumerate(problems):
        columns[p] = None
        columns[p] = copy.deepcopy(columns_prep)
        columns[p]["key"] = key_cols
        columns[p]["target"] = [target_cols[index]]
    problem = "3"
    cat_cols = columns[problem]["categoric"]
    numeric_cols = columns[problem]["numeric"]
    target = columns[problem]["target"]
    dflearning = dflearning[cat_cols + numeric_cols + target]
    X_train, X_test, y_train, y_test = train_test_split(dflearning.drop(target, axis=1), dflearning[target],
                                                        test_size=0.25, random_state=42)

    clear_stage = ClearNoCategoriesTransformer(categorical_cols=cat_cols)
    imputer = ImputeTransformer(numerical_cols=numeric_cols, categorical_cols=cat_cols)
    outliers = OutliersTransformer(numerical_cols=numeric_cols, categorical_cols=cat_cols)
    scale = ScalingTransformer(numerical_cols=numeric_cols)
    categorize = CategorizeByTargetTransformer(categorical_cols=cat_cols)
    correlations = CorrelationTransformer(numerical_cols=numeric_cols, categorical_cols=cat_cols, target=target,
                                          threshold=0.9)
    dummies = DummiesTransformer(cat_cols)

    steps_feat = [("clear_non_variance", clear_stage),
                  ("imputer", imputer),
                  ("outliers", outliers),
                  ("scaling", scale),
                  ("categorize", categorize),
                  ("correlations", correlations),
                  ("dummies", dummies)]
    # the pre process pipeline steps
    pipeline_feat = MLPipeline(steps=steps_feat)
    X_train = pipeline_feat.fit_transform(X_train, y_train).reset_index(drop=True)
    X_test = pipeline_feat.transform(X_test).reset_index(drop=True)
    clfs = [LogisticRegression(),
            MLPClassifier(activation="relu", hidden_layer_sizes=(100, 100), solver="sgd"),
            XGBClassifier(n_jobs=mp.cpu_count() - 2, objective="reg:logistic", scoring='roc_auc', bootstrap=True)]
    path = os.getcwd() + "/results/"
    X_train.to_csv(path + "x_train.csv", index=False)
    X_test.to_csv(path + "x_test.csv", index=False)
    y_train.to_csv(path + "y_train.csv", index=False)
    y_test.to_csv(path + "y_test.csv", index=False)

    for i, clf in enumerate(clfs):

        start_time = time.time()
        print("doing model {}".format(i))
        # Create the grid search hyperparameters
        params = {
            0: {
                'C': [10 ** -i for i in range(-3, 4)]},
            1: {'alpha': [10 ** -i for i in range(-3, 4)],
                'batch_size': [32, 64],
                'learning_rate_init': [10 ** -i for i in range(-3, 4)]},
            2: {'n_estimators': [800, 900, 1000],
                'colsample_bytree': [i / 10.0 for i in range(6, 9)],
                'max_depth': range(3, 6),
                'min_samples_split': [i / 10.0 for i in range(2, 6)],
                'min_samples_leaf': [i / 10.0 for i in range(2, 5)],
                'gamma': [0, 1, 5]}
        }
        if i == 1:
            grid = GridSearchCV(estimator=clf, param_grid=params[i], cv=3, refit=True, n_jobs=mp.cpu_count() - 2)
        else:
            grid = GridSearchCV(estimator=clf, param_grid=params[i], cv=3, refit=True, n_jobs=mp.cpu_count() - 2,
                                scoring="f1")
        grid.fit(X_train, y_train)
        train_score = grid.score(X_train, y_train)
        test_score = grid.score(X_test, y_test)
        pickle.dump(grid, open(path + "grid_{}.p".format(i), "wb"))
        print("problem {} results model {}".format(problem, i))
        print("train_score: {}".format(train_score))
        print("test_score: {}".format(test_score))
        print("the total for model {} in minutes is: {}".format(i, (time.time() - start_time) / 60))


def plot_roc(model, y_test, y_score):

    plt.figure()
    lw = 2
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve {}'.format(model))
    plt.legend(loc="lower right")
    plt.show()


def plot_hyper_params(grid):

    print(1)


def model_evaluation():

    path = os.getcwd() + "/results/"
    x_test = pd.read_csv(path + "x_test.csv")
    y_test = pd.read_csv(path + "y_test.csv")
    lr = pickle.load(open(path + "grid_0.p", "rb"))
    mlp = pickle.load(open(path + "grid_1.p", "rb"))
    xgboost = pickle.load(open(path + "grid_2.p", "rb"))
    print("lr best hyperparameters: {}".format(lr.best_params_))
    print("mlp best hyperparameters: {}".format(mlp.best_params_))
    print("xgboost best hyperparameters: {}".format(xgboost.best_params_))
    lr_p, lr_r, lr_t = precision_recall_curve(y_test, lr.predict_proba(x_test)[:, 1])
    mlp_p, mlp_r, mlp_t = precision_recall_curve(y_test, mlp.predict_proba(x_test)[:, 1])
    xgboost_p, xgboost_r, xgboost_t = precision_recall_curve(y_test, xgboost.predict_proba(x_test)[:, 1])
    lr_best_t, lr_f1, lr_precision, lr_recall, lr_accuracy = measures(lr_p, lr_r, lr_t, lr, x_test, y_test)
    mlp_best_t, mlp_f1, mlp_precision, mlp_recall, mlp_accuracy = measures(mlp_p, mlp_r, mlp_t, mlp, x_test, y_test)
    xgboost_best_t, xgboost_f1, xgboost_precision, xgboost_recall, xgboost_accuracy = measures(xgboost_p, xgboost_r, xgboost_t, xgboost, x_test, y_test)
    print("lr")
    print("best t: {}".format(lr_best_t))
    print("f1: {}".format(lr_f1))
    print("accuracy: {}".format(lr_precision))
    print("precision: {}".format(lr_precision))
    print("recall: {}".format(lr_recall))
    print("mlp")
    print("best t: {}".format(mlp_best_t))
    print("f1: {}".format(mlp_f1))
    print("accuracy: {}".format(mlp_precision))
    print("precision: {}".format(mlp_precision))
    print("recall: {}".format(mlp_recall))
    print("xgboost")
    print("best t: {}".format(xgboost_best_t))
    print("f1: {}".format(xgboost_f1))
    print("accuracy: {}".format(xgboost_precision))
    print("precision: {}".format(xgboost_precision))
    print("recall: {}".format(xgboost_recall))
    plot_roc("logistic regression", y_test, lr.predict_proba(x_test)[:, 1])
    plot_roc("Multilayer perceptron", y_test, mlp.predict_proba(x_test)[:, 1])
    plot_roc("xgboost", y_test, xgboost.predict_proba(x_test)[:, 1])


def evaluation():

    path = os.getcwd() + "/results/"
    X_train = pd.read_csv(path + "x_train.csv")
    lr = pickle.load(open(path + "grid_0.p", "rb")).best_estimator_
    df = pd.DataFrame([X_train.columns, lr.coef_[0], abs(lr.coef_[0])]).T
    df.columns = ["feature", "coefficient", "coefficient_abs"]
    df = df.sort_values("coefficient_abs",ascending=False)
    inte = lr.intercept_[0]
    df.append(pd.DataFrame([{"feature": "intercept", "coefficient": inte, "coefficient_abs": abs(inte)}]),
              sort=False).to_csv(path + "coeff.csv", index=False)


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


if __name__ == "__main__":

    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    start_time = time.time()
    # run()
    # model_evaluation()
    evaluation()
    print("the total time in minutes is: {}".format((time.time() - start_time) / 60))
