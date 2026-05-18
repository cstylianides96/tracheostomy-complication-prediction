from xml.parsers.expat import model
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GroupKFold
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from evaluation_multiclass import evaluate_multiclass

# Try to import StratifiedGroupKFold (available in scikit-learn >= 1.1) for stratified group CV
try:
    from sklearn.model_selection import StratifiedGroupKFold
    strat_group_available = True
except Exception:
    StratifiedGroupKFold = None
    strat_group_available = False
import joblib
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from evaluation import evaluate

# For dyn_full:
# accounts for groups by taking stayid column into account during gridsearch.fit()
# stratifiedGroupKfold does not use groups directly 
def create_model(model_name, df_name, hours_after_trach):
    results = pd.DataFrame(columns=['model', 'evaluation', 'best_params', 'train_auc_mean', 'train_auc_sd',
                                    'test_auc', 'test_auprc', 'test_brier',
                                    'youden_threshold', 'youden_sen', 'youden_spec', 'youden_prec', 'youden_npv', 'youden_acc',
                                    'sen90_threshold', 'sen90_sen', 'sen90_spec', 'sen90_prec', 'sen90_npv', 'sen90_acc'])

    df_train = pd.read_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_train_selected.csv')
    if df_name == 'dyn_full':
        # Expect stay_id preserved in train file for group-aware CV
        print(df_train['label'].value_counts(normalize=True, dropna=False))
        groups_train = df_train['stay_id'].values
        x_df_train = df_train.drop(columns=['stay_id', 'label'])
        print(x_df_train.shape)
        y_df_train = df_train['label']
    else:
        print(df_train.label.value_counts(normalize=True, dropna=False))
        x_df_train = df_train.iloc[:, :-1]
        print(x_df_train.shape)
        y_df_train = df_train.iloc[:, -1]

    scale_pos_weight = y_df_train.value_counts()[0]/y_df_train.value_counts()[1]



    if model_name == 'SVM': 
        param_grid = [
            {'C': [200],
            'kernel': ['linear'], #, 'rbf', 'poly'],
            'gamma': ['scale']}]
        model = SVC(probability=True, random_state=123, class_weight='balanced', decision_function_shape='ovr')

    elif model_name == 'RF':
        param_grid = [
            {'n_estimators': [40, 50, 60, 80, 100],
             'max_features': [0.7, 0.8, 0.9, 1],
             'max_samples': [0.8, 0.9, 1]}]
        model = RandomForestClassifier(random_state=123, class_weight='balanced')

    elif model_name == 'XGB':      
        param_grid = [
            {'n_estimators': [100, 150, 200, 250],  
             'eta': [0.0001, 0.001, 0.01, 0.1],
             'gamma': [0],
             'max_depth': [3, 4, 5],
             'subsample': [0.9, 1],
             'colsample_bytree': [0.8, 0.9, 1],
             'lambda':[0.0001, 0.001, 0.01, 0.1, 0.2],
            'alpha': [0.0001, 0.001, 0.01, 0.1]}]
        model = XGBClassifier(random_state=123, objective='multi:softprob', num_class=n_classes )


    # choose CV: stratified group-aware if dyn_full, otherwise StratifiedKFold
    if df_name == 'dyn_full':
        cv = StratifiedGroupKFold(n_splits=5)
    else:
        cv = StratifiedKFold(3)

    grid_search = RandomizedSearchCV(model, param_grid[0], cv=cv, scoring='roc_auc_ovr', random_state=123, n_iter=100, n_jobs=-1)
    if df_name == 'dyn_full':
        grid_search.fit(x_df_train, y_df_train, groups=groups_train)
    else:
        grid_search.fit(x_df_train, y_df_train)

    best_params = str(grid_search.best_params_)
    best_model = grid_search.best_estimator_
    cvres = grid_search.cv_results_

    for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
        print(mean_score, params)
    train_auc_mean = cvres['mean_test_score'][grid_search.best_index_]
    train_auc_sd = cvres['std_test_score'][grid_search.best_index_]

    # save model
    os.makedirs('models_' + str(hours_after_trach) + 'hrs', exist_ok=True)
    joblib.dump(best_model, 'models_' + str(hours_after_trach) + 'hrs/' +  model_name + '_' + df_name + '.pkl')

    # test set
    df_test = pd.read_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_test_selected.csv')

    print(df_test.label.value_counts(normalize=True, dropna=False))
    if df_name == 'dyn_full':
        x_df_test = df_test.drop(columns=['stay_id', 'label'])
    else:
        x_df_test = df_test.iloc[:, :-1]
    y_df_test = df_test.iloc[:, -1]
    print(x_df_test.shape)

    # predict on test set
    prob_all_classes = best_model.predict_proba(x_df_test)  # All classes
    n_classes = len(np.unique(y_df_train))
    
    # save raw timestep-level probabilities for dyn_full OR patient-level for others
    os.makedirs('predictions_' + str(hours_after_trach) + 'hrs', exist_ok=True)

    # Save all probabilities
    prob_df = pd.DataFrame(prob_all_classes, columns=[f'prob_class_{i}' for i in range(prob_all_classes.shape[1])])
    prob_df['actual'] = y_df_test.values
    prob_df.to_csv('predictions_' + str(hours_after_trach) + 'hrs/' + model_name + '_' + df_name + '_probabilities.csv', index=False)

    # Evaluate and save results based on df_name
    os.makedirs('results_' + str(hours_after_trach) + 'hrs', exist_ok=True)
    
    if df_name == 'dyn_full':
        # --- Timestep-level evaluation ---
        eval_window = evaluate_multiclass(
            prob_all_classes, 
            y_df_test.values, 
            n_classes,
            hours_after_trach=hours_after_trach,
            model_name=model_name,
            df_name=df_name,
            eval_type='multiclass')
        print(best_params)
        print("Timestep-level AUC:", eval_window['auc'])
        
        results.loc[len(results)] = [###################
            model_name, 'timestep-level', best_params, train_auc_mean, train_auc_sd,
            eval_window['auc'], eval_window['auprc'], eval_window['brier'],
            eval_window['threshold_youden'], eval_window['sen_youden'], eval_window['spec_youden'],
            eval_window['prec_youden'], eval_window['npv_youden'], eval_window['acc_youden'],
            eval_window['threshold_sen90'], eval_window['sen_sen90'], eval_window['spec_sen90'],
            eval_window['prec_sen90'], eval_window['npv_sen90'], eval_window['acc_sen90']
        ]
        
        # --- Patient-level evaluation (aggregate every 12 rows) ---
        group_id = np.arange(len(prob_df)) // 12
        agg = prob_df.groupby(group_id).agg({'prob': 'max', 'actual': 'max'}).reset_index(drop=True)
        patient_prob = agg['prob'].values
        patient_actual = agg['actual'].values #############################
                
        eval_patient =  evaluate_multiclass(
            prob_all_classes, 
            y_df_test.values, 
            n_classes,
            hours_after_trach=hours_after_trach,
            model_name=model_name,
            df_name=df_name,
            eval_type='multiclass')
        print("Patient-level AUC:", eval_patient['auc'])
        
        results.loc[len(results)] = [
            model_name, 'patient-level', best_params, train_auc_mean, train_auc_sd,
            eval_patient['auc'], eval_patient['auprc'], eval_patient['brier'],
            eval_patient['threshold_youden'], eval_patient['sen_youden'], eval_patient['spec_youden'],
            eval_patient['prec_youden'], eval_patient['npv_youden'], eval_patient['acc_youden'],
            eval_patient['threshold_sen90'], eval_patient['sen_sen90'], eval_patient['spec_sen90'],
            eval_patient['prec_sen90'], eval_patient['npv_sen90'], eval_patient['acc_sen90']
        ]
        
        # save aggregated probabilities (patient-level) for dyn_full #################
        agg[['prob']].to_csv('predictions_' + str(hours_after_trach) + 'hrs/' + model_name + '_' + df_name + '_probabilities_agg12.csv', index=False)
    
    else:
        # --- row-level (patient-level) evaluation for non-dynamic datasets ---
        eval_results =  evaluate_multiclass(
            prob_all_classes, 
            y_df_test.values, 
            n_classes,
            hours_after_trach=hours_after_trach,
            model_name=model_name,
            df_name=df_name,
            eval_type='multiclass')

        print(best_params)
        print("AUC:", eval_results['auc'])#####################
        
        results.loc[len(results)] = [#################
            model_name, 'patient-level', best_params, train_auc_mean, train_auc_sd,
            eval_results['auc'], eval_results['auprc'], eval_results['brier'],
            eval_results['threshold_youden'], eval_results['sen_youden'], eval_results['spec_youden'],
            eval_results['prec_youden'], eval_results['npv_youden'], eval_results['acc_youden'],
            eval_results['threshold_sen90'], eval_results['sen_sen90'], eval_results['spec_sen90'],
            eval_results['prec_sen90'], eval_results['npv_sen90'], eval_results['acc_sen90']
        ]

    # save/append results
    if not os.path.isfile('results_' + str(hours_after_trach) + 'hrs/' + df_name + '_results.csv'):
        results.to_csv('results_' + str(hours_after_trach) + 'hrs/' + df_name + '_results.csv', index=False)
    else:
        results.to_csv('results_' + str(hours_after_trach) + 'hrs/' + df_name + '_results.csv', mode='a', header=False, index=False)

def create_ml_models(): 
    models = ['SVM'] # ['SVM', 'RF', 'XGB']

    for model in models:
        create_model(model, 'latest_full', hours_after_trach=24) # 'dyn_full', 'agg_full', 'latest_full', 'flat_full', 'agg_latest_full', 'agg_flatten_full'
create_ml_models()
