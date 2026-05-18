from pyexpat import model
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.multioutput import MultiOutputClassifier
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def add_multilabel_outcome():
    """
    Add multilabel outcome to all processed datasets.
    """
    hours_list = [12, 24]
    datasets = ['dyn_full', 'agg_full', 'latest_full', 'flat_full', 'agg_latest_full', 'agg_flatten_full']
    
    for hours_after_trach in hours_list:
        for dataset in datasets:
            df = pd.read_csv(f'data_processed_{hours_after_trach}hrs/{dataset}.csv')
            print(df.iloc[:, -1])
            df = df.drop(columns=['label'], errors='ignore')
            cohort_types = pd.read_csv('data_processed/cohort_with_complication_types.csv')[['stay_id', 'complication_bleeding','complication_infection','complication_mechanical','complication_other']]
           
            df_merged = df.merge(cohort_types, on='stay_id', how='inner')
            df_merged.to_csv(f'data_processed_{hours_after_trach}hrs/{dataset}_multilabel.csv', index=False)
            print(f"\nMissing values in {dataset}:")
            print(df_merged.isnull().sum())
            print(f"Total missing: {df_merged.isnull().sum().sum()}")
            print(len(df_merged))


def create_train_test_split(hours_after_trach):
    label_cols = ['complication_bleeding', 'complication_infection', 'complication_mechanical', 'complication_other']
    
    for df_name in ['dyn_full', 'agg_full', 'latest_full', 'flat_full', 'agg_latest_full', 'agg_flatten_full']:
        print(f"Creating train-test split for {df_name} dataset (multilabel)")
        df = pd.read_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_multilabel.csv')

        # For dynamic (temporal) dataset, split at stay_id (group) level with multilabel stratification
        if df_name == 'dyn_full':
            # Get one set of labels per stay to perform stratified split on stays (avoid window leakage)
            group_df = df.groupby('stay_id')[label_cols].first().reset_index()
            stays = group_df['stay_id'].values
            stay_labels = group_df[label_cols].values  # Shape: (n_stays, n_labels)
            
            # Create dummy X for stay_ids (iterative_train_test_split requires X input)
            X_stays = stays.reshape(-1, 1)
            
            # Use iterative_train_test_split for multilabel stratification
            X_train_stays, y_train_labels, X_test_stays, y_test_labels = iterative_train_test_split(
                X_stays, 
                stay_labels, 
                test_size=0.2
            )
            
            train_stays = X_train_stays.flatten()
            test_stays = X_test_stays.flatten()

            # Keep stay_id in train/test for group-aware cross-validation downstream
            df_train = df[df['stay_id'].isin(train_stays)].reset_index(drop=True)
            df_test = df[df['stay_id'].isin(test_stays)].reset_index(drop=True)

            # Print label distribution
            print("Train label distribution:")
            print(df_train[label_cols].sum())
            print("\nTest label distribution:")
            print(df_test[label_cols].sum())

        else:
            # Row-level stratified shuffle split with multilabel stratification
            # Prepare X (features) and y (multilabel matrix)
            feature_cols = [col for col in df.columns if col not in ['stay_id'] + label_cols]
            X = df[feature_cols].values
            y = df[label_cols].values
            
            # Use iterative_train_test_split for multilabel stratification
            X_train, y_train, X_test, y_test = iterative_train_test_split(
                X, 
                y, 
                test_size=0.2
            )
            
            # Reconstruct dataframes
            df_train = pd.DataFrame(X_train, columns=feature_cols)
            df_train[label_cols] = y_train
            
            df_test = pd.DataFrame(X_test, columns=feature_cols)
            df_test[label_cols] = y_test
            
            # Print label distribution
            print("Train label distribution:")
            print(df_train[label_cols].sum())
            print("\nTest label distribution:")
            print(df_test[label_cols].sum())

        df_train.to_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_train_multilabel.csv', index=False)
        df_test.to_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_test_multilabel.csv', index=False)


def impute(hours_after_trach, df_name): 
    label_cols = ['complication_bleeding', 'complication_infection', 'complication_mechanical', 'complication_other']
    
    print(f"\n{df_name} - Multilabel Imputation")
    df_train = pd.read_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_train_multilabel.csv')
    df_test = pd.read_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_test_multilabel.csv')

    if df_name == 'dyn_full':
        # Drop stay_id and all label columns for features
        X_train = df_train.drop(columns=['stay_id'] + label_cols)
        y_train = df_train[label_cols].values  # Multilabel matrix
        id_train = df_train['stay_id']
        X_test = df_test.drop(columns=['stay_id'] + label_cols)
        y_test = df_test[label_cols].values
        id_test = df_test['stay_id']
    else:
        # Drop all label columns for features
        X_train = df_train.drop(columns=label_cols)
        y_train = df_train[label_cols].values  # Multilabel matrix
        X_test = df_test.drop(columns=label_cols)
        y_test = df_test[label_cols].values

    print(f'No. of features before selection: {X_train.shape[1]}')
    print(f'Label matrix shape: {y_train.shape}')

    # IterativeImputer for all datasets (fit on train, apply to test)
    imputer = IterativeImputer(
        random_state=123,
        max_iter=10,
        sample_posterior=False,
        skip_complete=True
    )
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=imputer.feature_names_in_)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=imputer.feature_names_in_)

    if df_name == 'dyn_full':
        return X_train, y_train, X_test, y_test, id_train, id_test
    else:
        return X_train, y_train, X_test, y_test


def feature_selection(hours_after_trach):

    if hours_after_trach ==12:
        X_train_dyn, y_train_dyn, X_test_dyn, y_test_dyn, id_train, id_test = impute(hours_after_trach, 'dyn_full')

        # Feature selection on dyn_full only
        # Variance threshold (fit on train, apply to test)
        sel = VarianceThreshold(threshold=0.05)
        X_train_dyn = pd.DataFrame(sel.fit_transform(X_train_dyn), columns=sel.get_feature_names_out())
        X_test_dyn = pd.DataFrame(sel.transform(X_test_dyn), columns=sel.get_feature_names_out())
        print('Dynamic dataset: No. of features after selection (variance):', X_train_dyn.shape[1])

        # Train Random Forest and get feature importances
        rf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=123, class_weight='balanced', n_jobs=-1))
        rf.fit(X_train_dyn, y_train_dyn)

        # Get feature importances sorted in descending order
        importances = np.mean([est.feature_importances_ for est in rf.estimators_], axis=0)
        feature_names = X_train_dyn.columns
        sorted_idx = np.argsort(importances)[::-1]
        sorted_importances = importances[sorted_idx]

        # Create elbow plot
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(sorted_importances)), np.cumsum(sorted_importances), 'bo-', linewidth=2, markersize=4)
        plt.xlabel('Number of Features')
        plt.ylabel('Cumulative Importance')
        plt.title('Feature Importance Elbow Plot')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'data_processed_{hours_after_trach}hrs/feature_importance_elbow_multilabel.png', dpi=300)
        plt.show()

        print(f"Top 16 features by importance:")
        for i in range(min(16, len(sorted_idx))):
            print(f"{i+1}. {feature_names[sorted_idx[i]]}: {sorted_importances[i]:.4f}")
        selected_features = feature_names[sorted_idx[:16]] # descending order
        print('SELECTED FEATURES:', selected_features)
        selected_features.to_series().to_csv(f'data_processed_{hours_after_trach}hrs/selected_features_dyn_full_multilabel.csv', index=False)

    else:  # for 24 hours, load selected features from 12 hours
        X_train_dyn, y_train_dyn, X_test_dyn, y_test_dyn, id_train, id_test = impute(hours_after_trach, 'dyn_full')
        
        # Load selected features from 12 hours
        selected_features = pd.read_csv(f'data_processed_12hrs/selected_features_dyn_full_multilabel.csv', header=None).squeeze().tolist()[1:]
        print('SELECTED FEATURES from 12hrs:', selected_features)

    X_train_dyn = X_train_dyn[selected_features]
    X_test_dyn = X_test_dyn[selected_features]

    # Save selected features train and test
    label_cols = ['complication_bleeding', 'complication_infection', 'complication_mechanical', 'complication_other']
    y_train_dyn_df = pd.DataFrame(y_train_dyn, columns=label_cols)
    y_test_dyn_df = pd.DataFrame(y_test_dyn, columns=label_cols)
    df_train_dyn_sel = pd.concat([id_train.reset_index(drop=True), X_train_dyn.reset_index(drop=True), y_train_dyn_df.reset_index(drop=True)], axis=1)
    df_test_dyn_sel = pd.concat([id_test.reset_index(drop=True), X_test_dyn.reset_index(drop=True), y_test_dyn_df.reset_index(drop=True)], axis=1)
    
    df_train_dyn_sel.to_csv(f'data_processed_{hours_after_trach}hrs/dyn_full_train_selected_multilabel.csv', index=False)
    df_test_dyn_sel.to_csv(f'data_processed_{hours_after_trach}hrs/dyn_full_test_selected_multilabel.csv', index=False)

    for df_name in ['agg_full', 'latest_full', 'flat_full', 'agg_latest_full', 'agg_flatten_full']:

        X_train, y_train, X_test, y_test = impute(hours_after_trach, df_name)

        X_train = X_train[[col for col in X_train.columns if any(col.startswith(feat) for feat in selected_features)]]
        X_test = X_test[X_train.columns]
        label_cols = ['complication_bleeding', 'complication_infection', 'complication_mechanical', 'complication_other']
        y_train_df = pd.DataFrame(y_train, columns=label_cols)
        y_test_df = pd.DataFrame(y_test, columns=label_cols)
        df_train_sel = pd.concat([X_train.reset_index(drop=True), y_train_df.reset_index(drop=True)], axis=1)
        df_test_sel = pd.concat([X_test.reset_index(drop=True), y_test_df.reset_index(drop=True)], axis=1)

        print(f'Selected features for {df_name}:')
        print(X_train.columns)
        print(X_train.shape, X_test.shape)
        print(df_train_sel.columns)

        df_train_sel.to_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_train_selected_multilabel.csv', index=False)
        df_test_sel.to_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_test_selected_multilabel.csv', index=False)


def check_missing_values(hours_after_trach):
    for df_name in ['dyn_full', 'agg_full', 'latest_full', 'flat_full', 'agg_latest_full', 'agg_flatten_full']:
        print(f"\n{df_name}:")
        df_train_sel = pd.read_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_train_selected_multilabel.csv')
        df_test_sel = pd.read_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_test_selected_multilabel.csv')
        
        print(f"Train missing values:\n{df_train_sel.isnull().sum()}")
        print(f"\nTest missing values:\n{df_test_sel.isnull().sum()}")
        print(f"Total missing in train: {df_train_sel.isnull().sum().sum()}")
        print(f"Total missing in test: {df_test_sel.isnull().sum().sum()}")
        # print(df_train_sel.label.value_counts(normalize=True, dropna=False))
        # print(df_test_sel.label.value_counts(normalize=True, dropna=False))


def print_label_distribution_before_after(hours_after_trach, df_name):
    """
    Print label distributions (counts and proportions) before and after balancing.
    Expects *_train_selected_multilabel.csv and *_train_selected_balanced_multilabel.csv
    """
    label_cols = ['complication_bleeding', 'complication_infection', 'complication_mechanical', 'complication_other']

    df_before = pd.read_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_train_selected_multilabel.csv')
    df_after = pd.read_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_train_selected_balanced_multilabel.csv')
    # df_before = pd.read_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_test_selected_multilabel.csv')
    
    print(f"\n{df_name} (hours_after_trach={hours_after_trach})")
    print("Before balancing (counts):")
    print(df_before[label_cols].sum())
    print("Before balancing (proportions):")
    print(df_before[label_cols].mean())

    print("\nAfter balancing (counts):")
    print(df_after[label_cols].sum())
    print("After balancing (proportions):")
    print(df_after[label_cols].mean())


def _mlsmote(X, y, k_neighbors=5, n_samples=None, random_state=123):
    """
    Basic multilabel-aware SMOTE implementation (ML-SMOTE style).
    - Generates synthetic samples by interpolating minority-label samples.
    - Assigns labels by OR-ing the labels of the seed and neighbor.
    """
    rng = np.random.default_rng(random_state)

    if n_samples is None:
        # default: balance each label up to the max positive count
        label_counts = y.sum(axis=0)
        target = int(np.max(label_counts))
        deficits = target - label_counts
        deficits = deficits[deficits > 0]
        n_samples = int(np.sum(deficits))

    if n_samples <= 0:
        return X, y

    # Identify minority samples (any label below mean)
    label_counts = y.sum(axis=0)
    minority_labels = label_counts < np.mean(label_counts)
    if not np.any(minority_labels):
        return X, y

    minority_mask = (y[:, minority_labels] == 1).any(axis=1)
    X_min = X[minority_mask]
    y_min = y[minority_mask]

    if X_min.shape[0] < 2:
        return X, y

    k = min(k_neighbors, X_min.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    nn.fit(X_min)
    neighbors = nn.kneighbors(X_min, return_distance=False)[:, 1:]

    synthetic_X = []
    synthetic_y = []

    for _ in range(n_samples):
        idx = rng.integers(0, X_min.shape[0])
        neigh_idx = rng.choice(neighbors[idx])

        x_i = X_min[idx]
        x_n = X_min[neigh_idx]
        y_i = y_min[idx]
        y_n = y_min[neigh_idx]

        delta = rng.random()
        x_new = x_i + delta * (x_n - x_i)
        y_new = np.where((y_i + y_n) > 0, 1, 0)

        synthetic_X.append(x_new)
        synthetic_y.append(y_new)

    X_syn = np.vstack([X, np.array(synthetic_X)])
    y_syn = np.vstack([y, np.array(synthetic_y)])

    return X_syn, y_syn


def apply_mlsmote(hours_after_trach, df_name, k_neighbors=5, n_samples=None, random_state=123):
    """
    Apply multilabel-aware SMOTE to the selected train set only.
    Writes *_train_selected_balanced_multilabel.csv
    """
    label_cols = ['complication_bleeding', 'complication_infection', 'complication_mechanical', 'complication_other']

    df_train = pd.read_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_train_selected_multilabel.csv')

    if df_name == 'dyn_full':
        id_col = df_train['stay_id']
        X = df_train.drop(columns=['stay_id'] + label_cols).values
    else:
        id_col = None
        X = df_train.drop(columns=label_cols).values

    y = df_train[label_cols].values

    X_bal, y_bal = _mlsmote(
        X,
        y,
        k_neighbors=k_neighbors,
        n_samples=n_samples,
        random_state=random_state
    )

    df_X = pd.DataFrame(X_bal, columns=df_train.drop(columns=(['stay_id'] if df_name == 'dyn_full' else []) + label_cols).columns)
    df_y = pd.DataFrame(y_bal, columns=label_cols)

    if df_name == 'dyn_full':
        # For synthetic rows, set stay_id to -1
        n_new = df_X.shape[0] - len(id_col)
        id_bal = pd.concat([id_col.reset_index(drop=True), pd.Series([-1] * n_new, name='stay_id')], axis=0)
        df_bal = pd.concat([id_bal.reset_index(drop=True), df_X.reset_index(drop=True), df_y.reset_index(drop=True)], axis=1)
    else:
        df_bal = pd.concat([df_X.reset_index(drop=True), df_y.reset_index(drop=True)], axis=1)

    df_bal.to_csv(
        f'data_processed_{hours_after_trach}hrs/{df_name}_train_selected_balanced_multilabel.csv',
        index=False
    )

    print(f"ML-SMOTE applied to {df_name} (train). New shape: {df_bal.shape}")


def preprocess_multilabel(hours_after_trach):
    add_multilabel_outcome()
    create_train_test_split(hours_after_trach)
    feature_selection(hours_after_trach)
    check_missing_values(hours_after_trach)
   
    for df_name in ['dyn_full', 'agg_full', 'latest_full', 'flat_full', 'agg_latest_full', 'agg_flatten_full']:
        apply_mlsmote(
            hours_after_trach,
            df_name,
            k_neighbors=5,
            n_samples=None,
            random_state=123
        )
        print_label_distribution_before_after(hours_after_trach, df_name)
