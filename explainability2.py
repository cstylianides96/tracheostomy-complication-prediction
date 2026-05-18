# SHAP beeswarm plot - not group aware (timestep-level) - 1 plot for test set (mean absolute SHAP values averaged across labels for each feature for multilabel)
from xml.parsers.expat import model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap 

LABEL_COLS = [
    'complication_bleeding',
    'complication_infection',
    'complication_mechanical',
    'complication_other'
]


def plot_shap2(model_name, df_name, hours_after_trach, multilabel):
    if multilabel:
        model = joblib.load('models_' + str(hours_after_trach) + 'hrs/' +  model_name + '_' + df_name + '_multilabel.pkl')
    else:
        model = joblib.load('models_' + str(hours_after_trach) + 'hrs/' +  model_name + '_' + df_name + '.pkl')

    model_feats = model.feature_names_in_.tolist()
    items = model_feats
    print(items)
    print(len(items))

    itemids = pd.read_csv('data_raw_mimic2.2/d_items.csv')[['itemid', 'label']]
    #icd10_codes = pd.read_csv('./utils/mappings/ICD9_to_ICD10_mapping.txt', sep='\t', dtype=str)[['icd10cm', 'diagnosis_description']]
    icd10_codes = pd.read_csv('./utils/icd10cm_codes_2024.csv', dtype=str)[['icd10_code', 'label']]

    labels = []
    for item in items:
        if ('_' in item) and (item[-1].isdigit()):  # ending in window number
            itemid = item.rsplit('_', 1)[0]
            label = itemids.loc[itemids['itemid'] == int(itemid)]['label'].values[0]
            label = label + '_' + item.rsplit('_', 1)[1]
            labels.append(label)
        elif ('_' in item) and (item.rsplit('_', 1)[1] in ['mean', 'median','std', 'min', 'max']): #stats for temporal features
            itemid = item.rsplit('_', 1)[0]
            label = itemids.loc[itemids['itemid'] == int(itemid)]['label'].values[0]
            label = label + '_' + item.rsplit('_', 1)[1]
            labels.append(label)
        elif item in icd10_codes['icd10_code'].tolist():  # diagnosis
            label = icd10_codes.loc[icd10_codes['icd10_code'] == item]['label'].values[0]
            labels.append(label)
        elif item.isdigit():  # hight, weight, drug, or features in dyn_full or latest_full
            itemid = int(item)
            label = itemids.loc[itemids['itemid'] == itemid]['label'].values[0]
            labels.append(label)
        else: # demographics
            labels.append(item)

    print(labels)

    if multilabel:
        df_train = pd.read_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_train_selected_balanced_multilabel.csv')
        df_test = pd.read_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_test_selected_multilabel.csv')
    else:
        df_train = pd.read_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_train_selected_balanced.csv')
        df_test = pd.read_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_test_selected.csv')

    if df_name == 'dyn_full': # not taking into account groups of windows, timestep level
        # Expect stay_id preserved in train file
        # print(df_train['label'].value_counts(normalize=True, dropna=False))
        groups_train = df_train['stay_id'].values
        x_df_train = df_train.drop(columns=['stay_id', 'complication_bleeding','complication_infection',
                                             'complication_mechanical', 'complication_other'])
        x_df_test = df_test.drop(columns=['stay_id', 'complication_bleeding','complication_infection',
                                             'complication_mechanical', 'complication_other'])
        print(x_df_train.shape)
       
    else:
        print(df_train.label.value_counts(normalize=True, dropna=False))
        x_df_train = df_train.iloc[:, :-1]
        x_df_test = df_test.iloc[:, :-1]
        print(x_df_train.shape)
       
    avg_shap_values = None

    if multilabel:
        x_df_train = x_df_train[model_feats]
        x_df_test = x_df_test[model_feats]
        shap_values_all = []
        base_values_all = []
        for idx, label_name in enumerate(LABEL_COLS):
            predict_fn = lambda X, i=idx: model.estimators_[i].predict_proba(X)[:, 1]
            explainer = shap.Explainer(predict_fn, x_df_train)
            shap_values = explainer(x_df_test)
            shap_values_all.append(shap_values)
            base_vals = shap_values.base_values
            if np.ndim(base_vals) == 0:
                base_vals = np.full((x_df_test.shape[0],), base_vals)
            base_values_all.append(base_vals)

        values_stack = np.stack([sv.values for sv in shap_values_all], axis=0)
        mean_abs_values = np.mean(np.abs(values_stack), axis=0)
        mean_base_values = np.mean(np.vstack(base_values_all), axis=0)
        avg_shap_values = shap.Explanation(
            values=mean_abs_values,
            base_values=mean_base_values,
            data=x_df_test.values,
            feature_names=labels
        )
        shap.plots.beeswarm(avg_shap_values, max_display=11)
        
    else:
        x_df_train = x_df_train[model_feats]
        x_df_test = x_df_test[model_feats]
        
        explainer = shap.Explainer(lambda X: model.predict_proba(X)[:, 1], x_df_train)
        shap_values = explainer(x_df_test)
        shap_values.feature_names = labels
        shap.plots.beeswarm(shap_values, max_display=11)  #plot_type="bar",
        # shap.plots.waterfall(shap_values[0], max_display=15)


    if multilabel:
        plt.savefig(f'plots_{hours_after_trach}hrs/{model_name}_{df_name}_shap_multilabel_avg.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f'plots_{hours_after_trach}hrs/{model_name}_{df_name}_shap.png', dpi=300, bbox_inches='tight')
