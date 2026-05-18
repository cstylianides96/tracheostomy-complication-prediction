import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from icu_preprocess_util import *
from outlier_removal import *
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt



def icd9_to_icd10(icd_code):
    """
    Convert ICD-9 code to ICD-10 code using the mapping file.
    
    Args:
        icd_code: ICD-9 code (string or numeric)
    
    Returns:
        ICD-10 code (string) or None if no mapping found
    """
    # Load the mapping file
    mapping_path = 'ICD9_to_ICD10_mapping.txt'
    
    if not hasattr(icd9_to_icd10, 'mapping_dict'):
        # Load mapping only once and cache it
        mapping_df = pd.read_csv(mapping_path, sep='\t', dtype=str)
        # Create dictionary mapping icd9cm to icd10cm
        icd9_to_icd10.mapping_dict = dict(zip(mapping_df['icd9cm'], mapping_df['icd10cm']))
    
    # Convert input to string and remove any leading/trailing whitespace
    icd_code_str = str(icd_code).strip()
    
    # Look up the ICD-10 code
    icd10_code = icd9_to_icd10.mapping_dict.get(icd_code_str, None)
    
    # Return None if the result is 'NoDx' (no diagnosis mapping available)
    if icd10_code == 'NoDx':
        return None
    
    return icd10_code


def detect_outliers_chart(): # for chart events

    chart = pd.read_csv("data_processed/trach_chartevents.csv")
    chart = outlier_imputation(chart, 'itemid', 'valuenum', 98, left_thresh=2, impute=True)
    chart.to_csv("data_processed/trach_chartevents.csv", index=False)


def make_equal_intervals(hours_after_trach, interval): # for chart events

    chart = pd.read_csv("data_processed/trach_chartevents.csv")
    chart['charttime'] = pd.to_datetime(chart['charttime'])

    cohort = pd.read_csv('data_processed/cohort.csv')
    cohort['trach_time'] = pd.to_datetime(cohort['trach_time'])

    # Keep only patients who survived at least hours_after_trach hours
    cohort = cohort[cohort['time_to_discharge_hours'] >= hours_after_trach]   #581 patients
    print(cohort)

    # Calculate time from tracheostomy in hours
    chart = chart.merge(cohort[['stay_id', 'trach_time']], on='stay_id', how='inner')
    chart['chart_from_trach'] = chart['charttime'] - chart['trach_time']
    chart['chart_from_trach'] = chart['chart_from_trach'].dt.total_seconds() / 3600  # in hours
    chart = chart.drop(columns=['trach_time'])
    print(chart.head())

    # Resampling in bins of size=interval
    final_chart = pd.DataFrame()
    
    for i in tqdm(range(1, hours_after_trach+1, interval)):

        # within interval: median of values, ignoring missing values
        sub_chart = (chart[(chart['chart_from_trach'] >= i) & (chart['chart_from_trach'] < i+interval)]
                     .groupby(['stay_id', 'itemid']).agg({'valuenum': np.nanmedian}).reset_index())
        sub_chart['chart_from_trach'] = i
        print(sub_chart)
        final_chart = pd.concat([final_chart, sub_chart], axis=0)  
    final_chart = final_chart.reset_index(drop=True)
    print(final_chart)

    return final_chart


def create_temporal_chart(hours_after_trach, interval): # for chart events

    # Resample chart events into equal intervals
    chart = make_equal_intervals(hours_after_trach, interval)

    cohort = pd.read_csv('data_processed/cohort.csv')
    # Keep only patients who survived at least hours_after_trach hours
    cohort = cohort[cohort['time_to_discharge_hours'] >= hours_after_trach]   #581 patients
    print(cohort)
    stayids = cohort['stay_id'].tolist()
    print(stayids)
    full_chart = []

    # Reshape to have one row per hour after trach per patient, impute across intervals
    for stayid in stayids:
        df = chart[chart['stay_id'] == stayid]
        df = df.pivot_table(index='chart_from_trach', columns='itemid', values='valuenum')
        print(df)
        add_indices = pd.Index(range(1, hours_after_trach + 1)).difference(df.index)
        print(add_indices)
        add_df = pd.DataFrame(index=add_indices, columns=df.columns).fillna(np.nan)
        print(add_df)
        df = pd.concat([df, add_df])
        print(df)
        df = df.sort_index()
        print(df)
  
        df = df.ffill()
        df = df.bfill()
        df = df.fillna(df.median())  # ffill, bfill, fill with median for missing chart values
        print(df)
        full_chart.append(df)

    full_chart = pd.concat(full_chart, ignore_index=True)
    print(full_chart)
    print(full_chart.isnull().sum()) # missing values here means item not recorded at all during X hours after trach for the patient
    
    # Drop features with more than 40% missing values
    missing_percent = (full_chart.isnull().sum() / len(full_chart)) * 100
    print(missing_percent)
    full_chart = full_chart.loc[:, missing_percent < 40]
    print(full_chart)
    stay_id_repeated = []
    for stayid in stayids:
        stay_id_repeated.extend([stayid] * hours_after_trach)
    full_chart.insert(0, 'stay_id', stay_id_repeated)
    
    os.makedirs('data_processed_' + str(hours_after_trach) + 'hrs', exist_ok=True)
    full_chart.to_csv('data_processed_' + str(hours_after_trach) + 'hrs/trach_chartevents_resampled.csv', index=False)

def create_static_drug(hours_after_trach): # for drug data
    
    drug = pd.read_csv('data_processed/trach_inputevents.csv')
    drug['starttime'] = pd.to_datetime(drug['starttime'])

    cohort = pd.read_csv('data_processed/cohort.csv')
    cohort['trach_time'] = pd.to_datetime(cohort['trach_time'])

    # Keep only patients who survived at least hours_after_trach hours
    cohort = cohort[cohort['time_to_discharge_hours'] >= hours_after_trach].reset_index(drop=True)   #581 patients

    # Calculate start time from tracheostomy in hours
    drug = drug.merge(cohort[['stay_id', 'trach_time']], on='stay_id', how='inner')
    drug['start_from_trach'] = drug['starttime'] - drug['trach_time']
    drug['start_from_trach'] = drug['start_from_trach'].dt.total_seconds() / 3600

    # Feature 225798: Vancomycin administered X hours after trach (binary)
    drug_filtered = drug[(drug['amount'] == 1) & (drug['start_from_trach'] <= hours_after_trach)]
    drug_filtered = drug_filtered[['stay_id']].drop_duplicates()
    drug_filtered['225798'] = 1
    print(drug_filtered)

    all_stay_ids = drug[['stay_id']].drop_duplicates()
    print(all_stay_ids)
    drug_static = all_stay_ids.merge(drug_filtered, on='stay_id', how='left')
    print(drug_static)
    drug_static['225798'] = drug_static['225798'].fillna(0).astype(int)
    print(drug_static)

    drug_static.to_csv('data_processed_' + str(hours_after_trach) + 'hrs/trach_drug_static.csv', index=False)

def create_static_diagnoses():

    cohort = pd.read_csv('data_processed/cohort.csv')[['subject_id', 'hadm_id', 'stay_id']]

    # Preprocess diagnoses with ICD-9 to ICD-10 conversion
    diag = pd.read_csv('data_processed/trach_diagnoses.csv')
    diag['icd_code_converted'] = diag.apply(
        lambda row: icd9_to_icd10(row['icd_code']) if row['icd_version'] == 9 else row['icd_code'], axis=1)
    diag = diag.drop(columns=['icd_code', 'icd_version', 'seq_num', 'icd_code_normalized', 'is_relevant'])
    diag = diag.rename(columns={'icd_code_converted': 'new_icd10_code'})
    diag = diag.dropna(subset=['new_icd10_code'])
    diag = diag.merge(cohort[['stay_id', 'subject_id', 'hadm_id']], on=['subject_id', 'hadm_id'], how='left')
    diag = diag.drop(columns=['subject_id', 'hadm_id'])

    # One-hot encode ICD-10 codes per stay_id
    diag = diag.pivot_table(index=['stay_id'], columns='new_icd10_code', values='new_icd10_code', aggfunc='size', fill_value=0)
    diag = (diag > 0).astype(int)
    diag = diag.reset_index()
    
    # Include all cohort stay_ids and fill missing diagnoses with 0 (patient with no diagnose)
    all_stays = cohort[['stay_id']].drop_duplicates()
    diag = all_stays.merge(diag, on='stay_id', how='left')
    diag = diag.fillna(0).astype(int)
    diag.to_csv('data_processed/trach_diagnoses_static.csv', index=False)



def create_static_demographics():
    demo = pd.read_csv('data_processed/trach_demographics.csv')

    # One-hot encode gender and race
    demo = pd.get_dummies(demo, columns=['gender', 'race'], prefix=['gender', 'race'])

    # Rename height and weight columns
    demo = demo.rename(columns={'226512': 'weight_on_adm', '226730': 'height'})

    demo.to_csv('data_processed/trach_demographics_static.csv', index=False)


def handle_temporal(hours_after_trach):#try sliding window later
    
    # dynamic temporal data
    df = pd.read_csv('data_processed_' + str(hours_after_trach) + 'hrs/trach_chartevents_resampled.csv')
    
    # aggregated data
    agg_df = df.groupby('stay_id').agg(['mean', 'median','std', 'min', 'max'])
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    agg_df = agg_df.reset_index()
   
    # latest value
    latest_df = df.groupby('stay_id').last().reset_index()
    
    # flattened data
    flat_df = df.copy()
    flat_df['row_num'] = flat_df.groupby('stay_id').cumcount() + 1
    flat_df = flat_df.pivot_table(index='stay_id', columns='row_num', aggfunc='first')
    flat_df.columns = ['_'.join(map(str, col)).strip() for col in flat_df.columns.values]
    flat_df = flat_df.reset_index()
  
    # aggregate + latest
    agg_latest = pd.concat([agg_df, latest_df.drop(columns=['stay_id'])], axis=1)

    # aggregate + flatten
    agg_flatten = pd.concat([agg_df, flat_df.drop(columns=['stay_id'])], axis=1)
    
    return df, agg_df, latest_df, flat_df, agg_latest, agg_flatten

def create_dataset(hours_after_trach): 
    dyn, agg_df, latest_df, flat_df, agg_latest, agg_flatten = handle_temporal(hours_after_trach)

    # Load static data
    drug = pd.read_csv('data_processed_' + str(hours_after_trach) + 'hrs/trach_drug_static.csv')
    diag = pd.read_csv('data_processed/trach_diagnoses_static.csv')
    demo = pd.read_csv('data_processed/trach_demographics_static.csv').drop(columns=['subject_id'])
    label = pd.read_csv('data_processed/cohort.csv')[['stay_id', 'label']]
    static = drug.merge(diag, on='stay_id', how='left')
    static = static.merge(demo, on='stay_id', how='left')
    static = static.merge(label, on='stay_id', how='left')

    # items = pd.read_csv('data_raw_mimic2.2/d_items.csv', usecols=['itemid', 'label'])
    # itemids = df.columns[1:].to_list()
    # names = items[items['itemid'].isin([int(i) for i in itemids])]['label'].tolist()
    # print("Chart item names:", names) #['Heart Rate', 'Respiratory Rate', 'Creatinine (serum)', 'Glucose (serum)', 'Magnesium', 'Sodium (serum)', 'Paw High', 'Calcium non-ionized', 'Phosphorous', 'Potassium (serum)']

    dyn_merged = dyn.merge(static, on='stay_id', how='left')
    print(dyn_merged.label.value_counts(normalize=True, dropna=False))
    dyn_merged.to_csv('data_processed_' + str(hours_after_trach) + 'hrs/dyn_full.csv', index=False)

    agg_df_merged = agg_df.merge(static, on='stay_id', how='left')
    print(agg_df_merged.label.value_counts(normalize=True, dropna=False))
    agg_df_merged.to_csv('data_processed_' + str(hours_after_trach) + 'hrs/agg_full.csv', index=False)

    latest_df_merged = latest_df.merge(static, on='stay_id', how='left')
    print(latest_df_merged.label.value_counts(normalize=True, dropna=False))
    latest_df_merged.to_csv('data_processed_' + str(hours_after_trach) + 'hrs/latest_full.csv', index=False)

    flat_df_merged = flat_df.merge(static, on='stay_id', how='left')
    print(flat_df_merged.label.value_counts(normalize=True, dropna=False))
    flat_df_merged.to_csv('data_processed_' + str(hours_after_trach) + 'hrs/flat_full.csv', index=False)

    agg_latest_merged = agg_latest.merge(static, on='stay_id', how='left')
    print(agg_latest_merged.label.value_counts(normalize=True, dropna=False))
    agg_latest_merged.to_csv('data_processed_' + str(hours_after_trach) + 'hrs/agg_latest_full.csv', index=False)

    agg_flatten_merged = agg_flatten.merge(static, on='stay_id', how='left')
    print(agg_flatten_merged.label.value_counts(normalize=True, dropna=False))
    agg_flatten_merged.to_csv('data_processed_' + str(hours_after_trach) + 'hrs/agg_flatten_full.csv', index=False)

def create_train_test_split(hours_after_trach):
    for df_name in ['dyn_full', 'agg_full', 'latest_full', 'flat_full', 'agg_latest_full', 'agg_flatten_full']:
        print(f"Creating train-test split for {df_name} dataset")
        df = pd.read_csv(f'data_processed_{hours_after_trach}hrs/{df_name}.csv')

        # For dynamic (temporal) dataset, split at stay_id (group) level and keep stratification by label
        if df_name == 'dyn_full':
            # Get one label per stay and perform stratified split on stays to avoid leakage of windows
            group_df = df.groupby('stay_id')['label'].first().reset_index()
            stays = group_df['stay_id'].values
            stay_labels = group_df['label'].values

            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(sss.split(stays.reshape(-1, 1), stay_labels))
            train_stays = stays[train_idx]
            test_stays = stays[test_idx]

            # Keep stay_id in train/test for group-aware cross-validation downstream
            df_train = df[df['stay_id'].isin(train_stays)].reset_index(drop=True)
            df_test = df[df['stay_id'].isin(test_stays)].reset_index(drop=True)

            print(df_train['label'].value_counts(normalize=True, dropna=False))
            print(df_test['label'].value_counts(normalize=True, dropna=False))

        else:
            # Original row-level stratified shuffle split (drop stay_id)
            df_nostay = df.drop(columns=['stay_id'])
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            for train_index, test_index in sss.split(df_nostay.drop(columns=['label']), df_nostay['label']):  # same proportions of label1:label0 for each split
                df_train = df_nostay.loc[train_index].reset_index(drop=True)
                df_test = df_nostay.loc[test_index].reset_index(drop=True)
                print(df_train['label'].value_counts(normalize=True, dropna=False))
                print(df_test['label'].value_counts(normalize=True, dropna=False))

        df_train.to_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_train.csv', index=False)
        df_test.to_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_test.csv', index=False)


def impute(hours_after_trach, df_name):
    
    df_train = pd.read_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_train.csv')
    df_test = pd.read_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_test.csv')

    if df_name == 'dyn_full':
        X_train = df_train.drop(columns=['stay_id', 'label'])
        y_train = df_train['label']
        id_train = df_train['stay_id']
        X_test = df_test.drop(columns=['stay_id', 'label'])
        y_test = df_test['label']
        id_test = df_test['stay_id']
    else:
        X_train = df_train.iloc[:, :-1]
        y_train = df_train.iloc[:, -1]
        X_test = df_test.iloc[:, :-1]
        y_test = df_test.iloc[:, -1]

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


def feature_selection(hours_after_trach): # imputation of all datasets + feature selection (variance, importance) based on dyn_full selected features

    if hours_after_trach ==12:
        X_train_dyn, y_train_dyn, X_test_dyn, y_test_dyn, id_train, id_test = impute(hours_after_trach, 'dyn_full')

        # Feature selection on dyn_full only
        # Variance threshold (fit on train, apply to test)
        sel = VarianceThreshold(threshold=0.05)
        X_train_dyn = pd.DataFrame(sel.fit_transform(X_train_dyn), columns=sel.get_feature_names_out())
        X_test_dyn = pd.DataFrame(sel.transform(X_test_dyn), columns=sel.get_feature_names_out())
        print('Dynamic dataset: No. of features after selection (variance):', X_train_dyn.shape[1])

        # Train Random Forest and get feature importances
        rf = RandomForestClassifier(n_estimators=100, random_state=123, class_weight='balanced', n_jobs=-1)
        rf.fit(X_train_dyn, y_train_dyn)

        # Get feature importances sorted in descending order
        importances = rf.feature_importances_
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
        plt.savefig(f'data_processed_{hours_after_trach}hrs/feature_importance_elbow.png', dpi=300)
        plt.show()

        print(f"Top 16 features by importance:")
        for i in range(min(16, len(sorted_idx))):
            print(f"{i+1}. {feature_names[sorted_idx[i]]}: {sorted_importances[i]:.4f}")
        selected_features = feature_names[sorted_idx[:16]] # descending order
        print('SELECTED FEATURES:', selected_features)
        selected_features.to_series().to_csv(f'data_processed_{hours_after_trach}hrs/selected_features_dyn_full.csv', index=False)

    else:  # for 24 hours, load selected features from 12 hours
        X_train_dyn, y_train_dyn, X_test_dyn, y_test_dyn, id_train, id_test = impute(hours_after_trach, 'dyn_full')
        
        # Load selected features from 12 hours
        selected_features = pd.read_csv(f'data_processed_12hrs/selected_features_dyn_full.csv', header=None).squeeze().tolist()[1:]
        print('SELECTED FEATURES from 12hrs:', selected_features)

    X_train_dyn = X_train_dyn[selected_features]
    X_test_dyn = X_test_dyn[selected_features]

    # Save selected features train and test
    df_train_dyn_sel = pd.concat([id_train.reset_index(drop=True), X_train_dyn.reset_index(drop=True), y_train_dyn.reset_index(drop=True)], axis=1)
    df_test_dyn_sel = pd.concat([id_test.reset_index(drop=True), X_test_dyn.reset_index(drop=True), y_test_dyn.reset_index(drop=True)], axis=1)
    
    df_train_dyn_sel.to_csv(f'data_processed_{hours_after_trach}hrs/dyn_full_train_selected.csv', index=False)
    df_test_dyn_sel.to_csv(f'data_processed_{hours_after_trach}hrs/dyn_full_test_selected.csv', index=False)

    for df_name in ['agg_full', 'latest_full', 'flat_full', 'agg_latest_full', 'agg_flatten_full']:

        X_train, y_train, X_test, y_test = impute(hours_after_trach, df_name)

        X_train = X_train[[col for col in X_train.columns if any(col.startswith(feat) for feat in selected_features)]]
        X_test = X_test[X_train.columns]
        df_train_sel = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
        df_test_sel = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

        print(f'Selected features for {df_name}:')
        print(X_train.columns)
        print(X_train.shape, X_test.shape)
        print(df_train_sel.columns)

        df_train_sel.to_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_train_selected.csv', index=False)
        df_test_sel.to_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_test_selected.csv', index=False)


def class_balance(hours_after_trach):
    for df_name in ['dyn_full', 'agg_full', 'latest_full', 'flat_full', 'agg_latest_full', 'agg_flatten_full']:
        print(f"\n{df_name}:")
        df_train_sel = pd.read_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_train_selected.csv')
        print(f"Train class balance:\n{df_train_sel.label.value_counts(normalize=True, dropna=False)}")

        X_train = df_train_sel.drop(columns=['label'])
        y_train = df_train_sel['label']

        smote = SMOTE(random_state=123)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        df_train_balanced = pd.concat([pd.DataFrame(X_train_smote, columns=X_train.columns), 
                                        pd.Series(y_train_smote, name='label')], axis=1)

        print(f"After SMOTE class balance:\n{df_train_balanced.label.value_counts(normalize=True, dropna=False)}")
        df_train_balanced.to_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_train_selected_balanced.csv', index=False)




def check_missing_values(hours_after_trach):
    for df_name in ['dyn_full', 'agg_full', 'latest_full', 'flat_full', 'agg_latest_full', 'agg_flatten_full']:
        print(f"\n{df_name}:")
        df_train_sel = pd.read_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_train_selected.csv')
        df_test_sel = pd.read_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_test_selected.csv')
        
        print(f"Train missing values:\n{df_train_sel.isnull().sum()}")
        print(f"\nTest missing values:\n{df_test_sel.isnull().sum()}")
        print(f"Total missing in train: {df_train_sel.isnull().sum().sum()}")
        print(f"Total missing in test: {df_test_sel.isnull().sum().sum()}")
        # print(df_train_sel.label.value_counts(normalize=True, dropna=False))
        # print(df_test_sel.label.value_counts(normalize=True, dropna=False))


def preprocess(hours_after_trach):
    detect_outliers_chart(),
    create_temporal_chart(hours_after_trach, 1),  # 24 hours after trach, 1-hour intervals
    create_static_drug(hours_after_trach),
    create_static_diagnoses(),
    create_static_demographics()
    create_dataset(hours_after_trach),
    create_train_test_split(hours_after_trach),
    feature_selection(hours_after_trach),
    check_missing_values(hours_after_trach)
    class_balance(hours_after_trach)


def analyze_processed_data():
    """
    Read and analyze all CSV files in data_processed_12hrs and data_processed_24hrs directories.
    Output shape and column names for each file.
    """
    directories = ['data_processed_12hrs'] #, 'data_processed_24hrs']
    
    for directory in directories:
        print(f"\n{'='*80}")
        print(f"Analysis of {directory}")
        print(f"{'='*80}\n")
        
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist.\n")
            continue
        
        # Get all CSV files in the directory
        csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        
        if not csv_files:
            print(f"No CSV files found in {directory}.\n")
            continue
        
        # Sort files for consistent output
        csv_files.sort()
        
        for csv_file in csv_files:
            file_path = os.path.join(directory, csv_file)
            try:
                df = pd.read_csv(file_path)
                print(f"File: {csv_file}")
                print(f"Shape: {df.shape} (rows: {df.shape[0]}, columns: {df.shape[1]})")
                print(f"Columns: {list(df.columns)[0:5]} ... {list(df.columns)[-5:]}")
                print(f"{'-'*80}\n")
            except Exception as e:
                print(f"Error reading {csv_file}: {str(e)}\n")

