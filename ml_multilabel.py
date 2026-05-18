from xml.parsers.expat import model
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, hamming_loss, jaccard_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils.class_weight import compute_sample_weight
from skmultilearn.model_selection import IterativeStratification
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
import joblib
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from evaluation_multilabel import evaluate_multilabel
from sklearn.metrics import make_scorer

# For dyn_full:
# accounts for groups my taking max label per group (used in iterative stratification fit)
def create_model_multilabel(model_name, df_name, hours_after_trach):
    # Updated columns for multilabel metrics
    results = pd.DataFrame(columns=[
        'model', 'evaluation', 'best_params', 'train_hamming_mean', 'train_hamming_sd',
        'test_hamming', 'test_jaccard_macro', 'test_f1_macro', 'test_auc_macro', 'test_ap_macro', 'test_brier_macro',
        'test_auc_complication_bleeding', 'test_auc_complication_infection', 'test_auc_complication_mechanical', 'test_auc_complication_other',
        'test_f1_complication_bleeding', 'test_f1_complication_infection', 'test_f1_complication_mechanical', 'test_f1_complication_other',
        'test_precision_complication_bleeding', 'test_precision_complication_infection', 'test_precision_complication_mechanical', 'test_precision_complication_other',
        'test_recall_complication_bleeding', 'test_recall_complication_infection', 'test_recall_complication_mechanical', 'test_recall_complication_other',
        'test_specificity_complication_bleeding', 'test_specificity_complication_infection', 'test_specificity_complication_mechanical', 'test_specificity_complication_other',
        'test_npv_complication_bleeding', 'test_npv_complication_infection', 'test_npv_complication_mechanical', 'test_npv_complication_other',
        'test_brier_complication_bleeding', 'test_brier_complication_infection', 'test_brier_complication_mechanical', 'test_brier_complication_other'
    ])

    df_train = pd.read_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_train_selected_balanced_multilabel.csv')
    
    # Identify label columns (assuming last 4 columns are labels)
    label_cols = ['complication_bleeding','complication_infection','complication_mechanical','complication_other']
    
    if df_name == 'dyn_full':
        print("Label distribution per class:")
        for col in label_cols:
            print(f"{col}: {df_train[col].value_counts(normalize=True, dropna=False)}")
        groups_train = df_train['stay_id'].values
        x_df_train = df_train.drop(columns=['stay_id'] + label_cols)
        print(x_df_train.shape)
        y_df_train = df_train[label_cols].values  # Convert to numpy array
    else:
        print("Label distribution per class:")
        for col in label_cols:
            print(f"{col}: {df_train[col].value_counts(normalize=True, dropna=False)}")
        x_df_train = df_train.drop(columns=label_cols)
        print(x_df_train.shape)
        y_df_train = df_train[label_cols].values  # Convert to numpy array

    n_labels = y_df_train.shape[1]
    print(f"Number of labels: {n_labels}")

    # Model setup with MultiOutputClassifier 
    if model_name == 'SVM': 
        param_grid = [
            {'estimator__C': [10, 50, 100, 200],
             'estimator__kernel': ['linear', 'rbf'],
             'estimator__gamma': ['scale', 'auto']}
        ]
        base_model = SVC(probability=True, random_state=123)
        model = MultiOutputClassifier(base_model)

    elif model_name == 'RF':
        param_grid = [
            {'estimator__n_estimators': [120],
             'estimator__max_features': [1],
             'estimator__max_samples': [0.7],}
        ]
        base_model = RandomForestClassifier(random_state=123)
        model = MultiOutputClassifier(base_model)

    elif model_name == 'XGB':      
        param_grid = [
            {'estimator__n_estimators': [80, 100, 120], 
             'estimator__eta': [0.001, 0.01, 0.1, 0.2],
             'estimator__gamma': [0],
             'estimator__max_depth': [6, 7, 8],
             'estimator__subsample': [0.7, 0.8, 0.9],
             'estimator__colsample_bytree': [0.7, 0.8, 0.9],
             'estimator__lambda': [0.001, 0.01, 0.1, 0.2],
             'estimator__alpha': [0.0001, 0.001, 0.01, 0.1],}
        ]
        base_model = XGBClassifier(random_state=123)
        model = MultiOutputClassifier(base_model)

    elif model_name == 'KNN':
        param_grid = [
            {'estimator__n_neighbors': [3, 5, 8, 11]}]
        base_model = KNeighborsClassifier(metric='euclidean')
        model = MultiOutputClassifier(base_model)

    elif model_name == 'MLP':
        param_grid = [
            {'estimator__hidden_layer_sizes': [(128,), (128, 64), (128, 64, 32), (128, 64, 32, 8), (128, 64, 8), (128, 32), (128, 32, 8), (128, 8), (64,), (64, 32), (64, 32, 8), (64, 8), (32,), (32, 8), (8,), (128, 128), (64, 64), (32, 32), (128, 128, 64), (64, 64, 32), (128, 128, 128), (64, 64, 64), (32, 32, 32), (16, 16, 16)],
             'estimator__alpha': [0.001, 0.01, 0.1, 0.2, 0.3],
             'estimator__learning_rate_init': [0.001, 0.01, 0.1]}]
        base_model = MLPClassifier(random_state=123, solver='adam', activation='relu', max_iter=200, 
                              early_stopping=True, validation_fraction=0.1)
        model = MultiOutputClassifier(base_model)



    # Custom CV scoring for multilabel using Hamming Loss (lower is better, so negate)
    hamming_scorer = make_scorer(hamming_loss, greater_is_better=False)

    # Iterative Stratification for multilabel CV
    cv_splits = []
    if df_name == 'dyn_full':
        # For dynamic data, group by stay_id to avoid data leakage
        # Get unique stay_ids and their indices
        unique_stay_ids = np.unique(groups_train)
        n_stays = len(unique_stay_ids)
        
        # Create a mapping of stay_id to row indices
        stay_id_to_indices = {}
        for stay_id in unique_stay_ids:
            stay_id_to_indices[stay_id] = np.where(groups_train == stay_id)[0]
        
        # Use IterativeStratification on aggregated stay-level labels
        # Aggregate labels per stay_id (take max to indicate if complication occurred)
        stay_labels = []
        stay_ids_list = []
        for stay_id in unique_stay_ids:
            indices = stay_id_to_indices[stay_id]
            # Max aggregation: if complication occurred at any timestep, mark as 1
            agg_labels = y_df_train[indices].max(axis=0)
            stay_labels.append(agg_labels)
            stay_ids_list.append(stay_id)
        
        stay_labels = np.array(stay_labels)
        stay_ids_array = np.array(stay_ids_list)
        
        # Use IterativeStratification on stay-level labels
        k_fold = IterativeStratification(n_splits=5, order=1)
        for train_stay_idx, val_stay_idx in k_fold.split(stay_ids_array, stay_labels):
            # Convert stay indices to row indices
            train_stay_ids = stay_ids_array[train_stay_idx]
            val_stay_ids = stay_ids_array[val_stay_idx]
            
            # Get all row indices for these stay_ids
            train_row_indices = np.concatenate([stay_id_to_indices[sid] for sid in train_stay_ids])
            val_row_indices = np.concatenate([stay_id_to_indices[sid] for sid in val_stay_ids])
            
            cv_splits.append((train_row_indices, val_row_indices))
 
    else:
        k_fold = IterativeStratification(n_splits=3, order=1)
        for train_idx, val_idx in k_fold.split(x_df_train.values, y_df_train):
            cv_splits.append((train_idx, val_idx))

    # RandomizedSearchCV with custom CV splits
    grid_search = RandomizedSearchCV(
        model, 
        param_grid[0], 
        cv=cv_splits, 
        scoring=hamming_scorer,  # Use hamming loss for multilabel
        random_state=123, 
        n_iter=100, 
        n_jobs=-1
    )


    grid_search.fit(x_df_train, y_df_train)

    best_params = str(grid_search.best_params_)
    best_model = grid_search.best_estimator_
    cvres = grid_search.cv_results_

    for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
        print(mean_score, params)
    train_hamming_mean = -cvres['mean_test_score'][grid_search.best_index_]  # Negate back
    train_hamming_sd = cvres['std_test_score'][grid_search.best_index_]

    # save model
    os.makedirs('models_' + str(hours_after_trach) + 'hrs', exist_ok=True)
    joblib.dump(best_model, 'models_' + str(hours_after_trach) + 'hrs/' + model_name + '_' + df_name + '_multilabel.pkl')

    # test set
    df_test = pd.read_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_test_selected_multilabel.csv')

    print("Test label distribution per class:")
    for col in label_cols:
        print(f"{col}: {df_test[col].value_counts(normalize=True, dropna=False)}")
    
    if df_name == 'dyn_full':
        x_df_test = df_test.drop(columns=['stay_id'] + label_cols)
    else:
        x_df_test = df_test.drop(columns=label_cols)
    y_df_test = df_test[label_cols].values
    print(x_df_test.shape)

    # predict on test set - get probabilities for all labels
    # MultiOutputClassifier.predict_proba returns list of arrays
    prob_list = best_model.predict_proba(x_df_test)
    
    # Extract probability of positive class (class 1) for each label
    prob_all_labels = np.column_stack([prob_list[i][:, 1] for i in range(n_labels)])
    
    # Get binary predictions (threshold at 0.5)
    y_pred = (prob_all_labels > 0.5).astype(int)

    # save raw probabilities
    prob_df = pd.DataFrame(prob_all_labels, columns=[f'prob_label_{i}' for i in range(n_labels)])
    for i, col in enumerate(label_cols):
        prob_df[f'actual_{col}'] = y_df_test[:, i]
    
    os.makedirs('predictions_' + str(hours_after_trach) + 'hrs', exist_ok=True)
    prob_df.to_csv('predictions_' + str(hours_after_trach) + 'hrs/' + model_name + '_' + df_name + '_probabilities_multilabel.csv', index=False)

    # Evaluate multilabel metrics
    os.makedirs('results_' + str(hours_after_trach) + 'hrs', exist_ok=True)
    
    if df_name == 'dyn_full':
        # --- Timestep-level evaluation ---
        eval_window = evaluate_multilabel(
            prob_all_labels, 
            y_df_test, 
            n_labels,
            hours_after_trach=hours_after_trach,
            decimals=4,
            model_name=model_name,
            df_name=df_name,
            eval_type='timestep-level'
        )
        
        print(best_params)
        print("AUC-Macro:", eval_window['auc_macro'], eval_window['auc_per_label'])
        print("Timestep-level Hamming Loss:", eval_window['hamming_loss'])
        print("Timestep-level F1-Macro:", eval_window['f1_macro'])
        
        results.loc[len(results)] = [
            model_name, 'timestep-level', best_params, train_hamming_mean, train_hamming_sd,
            eval_window['hamming_loss'], eval_window['jaccard_macro'], eval_window['f1_macro'],
            eval_window['auc_macro'], eval_window['ap_macro'], eval_window['brier_macro'],
            eval_window['auc_per_label'][0], eval_window['auc_per_label'][1], 
            eval_window['auc_per_label'][2], eval_window['auc_per_label'][3],
            eval_window['f1_per_label'][0], eval_window['f1_per_label'][1],
            eval_window['f1_per_label'][2], eval_window['f1_per_label'][3],
            eval_window['precision_per_label'][0], eval_window['precision_per_label'][1],
            eval_window['precision_per_label'][2], eval_window['precision_per_label'][3],
            eval_window['recall_per_label'][0], eval_window['recall_per_label'][1],
            eval_window['recall_per_label'][2], eval_window['recall_per_label'][3],
            eval_window['specificity_per_label'][0], eval_window['specificity_per_label'][1],
            eval_window['specificity_per_label'][2], eval_window['specificity_per_label'][3],
            eval_window['npv_per_label'][0], eval_window['npv_per_label'][1],
            eval_window['npv_per_label'][2], eval_window['npv_per_label'][3],
            eval_window['brier_per_label'][0], eval_window['brier_per_label'][1],
            eval_window['brier_per_label'][2], eval_window['brier_per_label'][3]
        ]
        
        # --- Patient-level evaluation (aggregate every hours_after_trach rows) ---
        group_id = np.arange(len(prob_df)) // hours_after_trach
        # Aggregate: max probability and max actual for each label
        agg_data = {'group': group_id}
        for i in range(n_labels):
            agg_data[f'prob_label_{i}'] = prob_all_labels[:, i]
            agg_data[f'actual_label_{i}'] = y_df_test[:, i]
        
        agg_df = pd.DataFrame(agg_data)
        agg = agg_df.groupby('group').agg({
            **{f'prob_label_{i}': 'max' for i in range(n_labels)},
            **{f'actual_label_{i}': 'max' for i in range(n_labels)}
        }).reset_index(drop=True)
        
        patient_prob = agg[[f'prob_label_{i}' for i in range(n_labels)]].values
        patient_actual = agg[[f'actual_label_{i}' for i in range(n_labels)]].values
        
        eval_patient = evaluate_multilabel(
            patient_prob,
            patient_actual,
            n_labels,
            hours_after_trach=hours_after_trach,
            decimals=5,
            model_name=model_name,
            df_name=df_name,
            eval_type='patient-level'
        )

        print("AUC-Macro:", eval_patient['auc_macro'], eval_patient['auc_per_label'])
        print("Patient-level Hamming Loss:", eval_patient['hamming_loss'])
        print("Patient-level F1-Macro:", eval_patient['f1_macro'])
        
        results.loc[len(results)] = [
            model_name, 'patient-level', best_params, train_hamming_mean, train_hamming_sd,
            eval_patient['hamming_loss'], eval_patient['jaccard_macro'], eval_patient['f1_macro'],
            eval_patient['auc_macro'], eval_patient['ap_macro'], eval_patient['brier_macro'],
            eval_patient['auc_per_label'][0], eval_patient['auc_per_label'][1],
            eval_patient['auc_per_label'][2], eval_patient['auc_per_label'][3],
            eval_patient['f1_per_label'][0], eval_patient['f1_per_label'][1],
            eval_patient['f1_per_label'][2], eval_patient['f1_per_label'][3],
            eval_patient['precision_per_label'][0], eval_patient['precision_per_label'][1],
            eval_patient['precision_per_label'][2], eval_patient['precision_per_label'][3],
            eval_patient['recall_per_label'][0], eval_patient['recall_per_label'][1],
            eval_patient['recall_per_label'][2], eval_patient['recall_per_label'][3],
            eval_patient['specificity_per_label'][0], eval_patient['specificity_per_label'][1],
            eval_patient['specificity_per_label'][2], eval_patient['specificity_per_label'][3],
            eval_patient['npv_per_label'][0], eval_patient['npv_per_label'][1],
            eval_patient['npv_per_label'][2], eval_patient['npv_per_label'][3],
            eval_patient['brier_per_label'][0], eval_patient['brier_per_label'][1],
            eval_patient['brier_per_label'][2], eval_patient['brier_per_label'][3]
        ]
        
        # save aggregated probabilities (patient-level)
        agg[[f'prob_label_{i}' for i in range(n_labels)]].to_csv(
            'predictions_' + str(hours_after_trach) + 'hrs/' + model_name + '_' + df_name + '_probabilities_agg' + str(hours_after_trach) + '_multilabel.csv',
            index=False
        )
    
    else:
        # --- row-level (patient-level) evaluation for non-dynamic datasets ---
        eval_results = evaluate_multilabel(
            prob_all_labels,
            y_df_test,
            n_labels,
            hours_after_trach=hours_after_trach,
            decimals=5,
            model_name=model_name,
            df_name=df_name,
            eval_type='patient-level'
        )
        
        print(best_params)
        print("AUC-Macro:", eval_results['auc_macro'], eval_results['auc_per_label'])
        print("Hamming Loss:", eval_results['hamming_loss'])
        print("F1-Macro:", eval_results['f1_macro'])
        
        results.loc[len(results)] = [
            model_name, 'patient-level', best_params, train_hamming_mean, train_hamming_sd,
            eval_results['hamming_loss'], eval_results['jaccard_macro'], eval_results['f1_macro'],
            eval_results['auc_macro'], eval_results['ap_macro'], eval_results['brier_macro'],
            eval_results['auc_per_label'][0], eval_results['auc_per_label'][1],
            eval_results['auc_per_label'][2], eval_results['auc_per_label'][3],
            eval_results['f1_per_label'][0], eval_results['f1_per_label'][1],
            eval_results['f1_per_label'][2], eval_results['f1_per_label'][3],
            eval_results['precision_per_label'][0], eval_results['precision_per_label'][1],
            eval_results['precision_per_label'][2], eval_results['precision_per_label'][3],
            eval_results['recall_per_label'][0], eval_results['recall_per_label'][1],
            eval_results['recall_per_label'][2], eval_results['recall_per_label'][3],
            eval_results['specificity_per_label'][0], eval_results['specificity_per_label'][1],
            eval_results['specificity_per_label'][2], eval_results['specificity_per_label'][3],
            eval_results['npv_per_label'][0], eval_results['npv_per_label'][1],
            eval_results['npv_per_label'][2], eval_results['npv_per_label'][3],
            eval_results['brier_per_label'][0], eval_results['brier_per_label'][1],
            eval_results['brier_per_label'][2], eval_results['brier_per_label'][3]
        ]

    # save/append results
    if not os.path.isfile('results_' + str(hours_after_trach) + 'hrs/' + df_name + '_results_multilabel.csv'):
        results.to_csv('results_' + str(hours_after_trach) + 'hrs/' + df_name + '_results_multilabel.csv', index=False)
    else:
        results.to_csv('results_' + str(hours_after_trach) + 'hrs/' + df_name + '_results_multilabel.csv', mode='a', header=False, index=False)
