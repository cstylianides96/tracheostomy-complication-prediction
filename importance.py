import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


def plot_feat_importances(hours_after_trach, model_name, df_name, multilabel): 
    
    if multilabel:
        model = joblib.load('models_' + str(hours_after_trach) + 'hrs/' +  model_name + '_' + df_name + '_multilabel.pkl')
    else:
        model = joblib.load('models_' + str(hours_after_trach) + 'hrs/' +  model_name + '_' + df_name + '.pkl')
    model_feats = model.feature_names_in_.tolist()
    X_cols = model_feats
    print(len(X_cols))


    itemids = pd.read_csv('data_raw_mimic2.2/d_items.csv')[['itemid', 'label']]
    #icd10_codes = pd.read_csv('./utils/mappings/ICD9_to_ICD10_mapping.txt', sep='\t', dtype=str)[['icd10cm', 'diagnosis_description']]
    icd10_codes = pd.read_csv('./utils/icd10cm_codes_2024.csv', dtype=str)[['icd10_code', 'label']]

    if multilabel:
        importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
    else:
        importances = model.feature_importances_

    indices = np.argsort(importances) # least to most important
    items = [X_cols[i] for i in indices]
    #print(items)
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
    print(importances[indices])

    # Save (least to most important)
    labels_df = pd.DataFrame(labels, columns=['Feature'])
    importances_df = pd.DataFrame(importances, columns=['Importance'])
    labels_df.to_csv('plots_' + str(hours_after_trach) + 'hrs/' + model_name + '_' + df_name + '_feature_labels.csv', index=False)
    importances_df.to_csv('plots_' + str(hours_after_trach) + 'hrs/' + model_name + '_' + df_name + '_feature_importances.csv', index=False)
    
    
    # Calculate appropriate figure size based on number of features
    num_features = len(labels)
    
    # Adjust font size based on number of features
    font_size = max(6, min(10, 200 / num_features))

    # Filter for non-zero importances
    non_zero_mask = importances[indices] > 0
    labels = [labels[i] for i in range(len(labels)) if non_zero_mask[i]]
    filtered_importances = importances[indices][non_zero_mask]
    
    # Keep only top 10 features for plotting
    filtered_importances = filtered_importances[-10:]  # Keep top 10 features
    labels = labels[-10:]

    # Recalculate figure size for top 10 features - compact layout
    fig_height = 4
    fig_width = 8
    plt.figure(figsize=(fig_width, fig_height))
    plt.barh(labels, filtered_importances, color='b', align='center', height=0.6)
    plt.yticks(fontsize=9)
    plt.xticks(fontsize=9)
    plt.xlabel('Relative Importance', fontsize=10)
    plt.title('Feature Importances', fontsize=11)
    plt.tight_layout()

    # # Recalculate figure size based on filtered features
    # fig_height = max(8, num_features * 0.3)
    # fig_width = 8  # Reduced from 12
    # font_size = max(8, min(12, 200 / num_features))  # Increased minimum and maximum
    # plt.figure(figsize=(fig_width, fig_height))
    # plt.barh(labels, filtered_importances, color='b', align='center')
    # plt.yticks(fontsize=font_size)
    # plt.xticks(fontsize=10)
    # plt.xlabel('Relative Importance', fontsize=10)
    # plt.title('Feature Importances', fontsize=10)
    # plt.tight_layout()
    if multilabel:
        plt.savefig('plots_' + str(hours_after_trach) + 'hrs/' + model_name + '_' + df_name + '_feature_importances_multilabel_top10.png', dpi=150, bbox_inches='tight')
    else:
        plt.savefig('plots_' + str(hours_after_trach) + 'hrs/' + model_name + '_' + df_name + '_feature_importances_top10.png', dpi=150, bbox_inches='tight')
    plt.show()
