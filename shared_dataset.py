import pandas as pd
from data_preprocessing import impute
from data_preprocessing_multilabel import impute as impute_multilabel

# datasets after data engineering and imputation, before feature selection

def compl_status(): # complication status label (binary), cohort includes patients that underwent tracheostomy
    X_train, y_train, X_test, y_test = impute(12, 'agg_flatten_full')
    shared_dataset_train = pd.concat([X_train, y_train], axis=1)
    shared_dataset_test = pd.concat([X_test, y_test], axis=1)
    shared_dataset = pd.concat([shared_dataset_train, shared_dataset_test], axis=0)

    print(shared_dataset.shape)
    print(shared_dataset.head())
    print(shared_dataset.label.value_counts())

    shared_dataset.to_csv('shared_dataset.csv', index=False)

def compl_type(): # complication type label (multilabel), cohort includes patients that underwent tracheostomy and had complications
    LABEL_COLS_MULTILABEL = [
        'complication_bleeding',
        'complication_infection',
        'complication_mechanical',
        'complication_other',
    ]

    X_train, y_train, X_test, y_test = impute_multilabel(12, 'agg_flatten_full')

    # `y_train`/`y_test` are numpy arrays from multilabel preprocessing.
    y_train_df = pd.DataFrame(y_train, columns=LABEL_COLS_MULTILABEL)
    y_test_df = pd.DataFrame(y_test, columns=LABEL_COLS_MULTILABEL)

    shared_dataset_train = pd.concat([X_train.reset_index(drop=True), y_train_df.reset_index(drop=True)], axis=1)
    shared_dataset_test = pd.concat([X_test.reset_index(drop=True), y_test_df.reset_index(drop=True)], axis=1)
    shared_dataset = pd.concat([shared_dataset_train, shared_dataset_test], axis=0)

    print(shared_dataset.shape)
    print(shared_dataset.head())
    print(shared_dataset[LABEL_COLS_MULTILABEL].sum())

    shared_dataset.to_csv('shared_dataset_multilabel.csv', index=False)
