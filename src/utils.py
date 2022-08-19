import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score


def agg_numeric_features(df, id_column, features):
    df_agg = df[features + [id_column]].groupby(id_column, as_index = False).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()
    df_agg.columns = ['_'.join(col) for col in df_agg.columns.values if col!=id_column]
    df_agg = df_agg.rename(columns={f"{id_column}_": id_column})
    
    return df_agg


def agg_categorical_features(df, id_column, features):
    categorical = pd.get_dummies(df.select_dtypes('object'))
    categorical[id_column] = df[id_column]
    df_agg = categorical.groupby(id_column).agg(['count'])
    df_agg.columns = ['_'.join(col) for col in df_agg.columns.values if col!=id_column]
    df_agg = df_agg.rename(columns={f"{id_column}_": id_column})
    
    return df_agg


def agg_stats_data(df, id_column):
    CATEGORICAL_VARS = [var for var in df.columns if df[var].dtype=='O']
    print("Number categorical variables: {}".format(len(CATEGORICAL_VARS)))

    NUMERICAL_VARS = [var for var in df.columns if var not in CATEGORICAL_VARS and var!=id_column]
    print("Number numerical variables: {}".format(len(NUMERICAL_VARS)))
    
    if NUMERICAL_VARS and CATEGORICAL_VARS:
        df_agg = pd.merge(
            agg_numeric_features(df, id_column, NUMERICAL_VARS),
            agg_categorical_features(df, id_column, CATEGORICAL_VARS), 
            on=id_column, how='inner'
        )
    elif not NUMERICAL_VARS:
        df_agg = agg_categorical_features(df, id_column, CATEGORICAL_VARS)
    elif not CATEGORICAL_VARS:
        df_agg = agg_numeric_features(df, id_column, NUMERICAL_VARS)
        
    return df_agg


def split_with_stratified_shuffle_split(df, target, n_splits, test_size, random_state):

    X_train_aux = df.copy()
    y_train_aux = X_train_aux.pop(target)

    sssplit = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    
    for validation_index_train, validation_index_test in sssplit.split(X_train_aux, y_train_aux):
        print(len(validation_index_test.tolist()))

    mask = df.reset_index(drop=True).index.isin(validation_index_test.tolist())
    id_test = df[df.reset_index(drop=True).index.isin(validation_index_test.tolist())].index
    id_train = df[df.reset_index(drop=True).index.isin(validation_index_train.tolist())].index

    df_train = df[~df.index.isin(id_test)]
    df_test = df[~df.index.isin(id_train)]
    print('Data Train info: {}'.format(len(id_train)))
    print('Data Test info: {}'.format(len(id_test)))

    return df_train, df_test


def get_classifier_metrics(Y_train, y_train_scores, Y_validation, Y_validation_scores):

    return {
        "roc_auc_train": roc_auc_score(Y_train, y_train_scores),
        "pr_auc_train":average_precision_score(Y_train, y_train_scores),
        "roc_auc_validation": roc_auc_score(Y_validation, Y_validation_scores),
        "pr_auc_validation":average_precision_score(Y_validation, Y_validation_scores),
    }