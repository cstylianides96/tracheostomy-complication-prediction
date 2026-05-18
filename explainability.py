# SHAP beeswarm plot - group aware - 1 plot per label for multilabel, for both train and test (8 plots for multilabel)
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


def _get_model_features(model):
    if hasattr(model, 'feature_names_in_'):
        return model.feature_names_in_.tolist()
    if hasattr(model, 'estimators_') and len(model.estimators_) > 0 and hasattr(model.estimators_[0], 'feature_names_in_'):
        return model.estimators_[0].feature_names_in_.tolist()
    raise AttributeError('Model does not expose feature names.')


def _get_feature_labels(items):
    itemids = pd.read_csv('data_raw_mimic2.2/d_items.csv')[['itemid', 'label']]
    icd10_codes = pd.read_csv('./utils/icd10cm_codes_2024.csv', dtype=str)[['icd10_code', 'label']]

    labels = []
    for item in items:
        if ('_' in item) and (item[-1].isdigit()):  # ending in window number
            itemid = item.rsplit('_', 1)[0]
            label = itemids.loc[itemids['itemid'] == int(itemid)]['label'].values[0]
            label = label + '_' + item.rsplit('_', 1)[1]
            labels.append(label)
        elif ('_' in item) and (item.rsplit('_', 1)[1] in ['mean', 'median', 'std', 'min', 'max']):
            itemid = item.rsplit('_', 1)[0]
            label = itemids.loc[itemids['itemid'] == int(itemid)]['label'].values[0]
            label = label + '_' + item.rsplit('_', 1)[1]
            labels.append(label)
        elif item in icd10_codes['icd10_code'].tolist():
            label = icd10_codes.loc[icd10_codes['icd10_code'] == item]['label'].values[0]
            labels.append(label)
        elif item.isdigit():
            itemid = int(item)
            label = itemids.loc[itemids['itemid'] == itemid]['label'].values[0]
            labels.append(label)
        else:
            labels.append(item)
    return labels


def _aggregate_by_group(shap_values, x_df, groups):
    group_series = pd.Series(groups, name='stay_id')
    x_df = x_df.reset_index(drop=True)
    grouped_data = x_df.assign(stay_id=group_series.values).groupby('stay_id').mean()

    shap_df = pd.DataFrame(shap_values.values, columns=x_df.columns)
    grouped_shap = shap_df.assign(stay_id=group_series.values).groupby('stay_id').mean()

    base_values = shap_values.base_values
    if np.ndim(base_values) == 0:
        grouped_base = np.full((grouped_shap.shape[0],), base_values)
    else:
        base_df = pd.DataFrame({'base_values': base_values, 'stay_id': group_series.values})
        grouped_base = base_df.groupby('stay_id')['base_values'].mean().values

    return shap.Explanation(
        values=grouped_shap.values,
        base_values=grouped_base,
        data=grouped_data.values,
        feature_names=list(x_df.columns)
    )


def _load_data(df_name, hours_after_trach, multilabel):
    if multilabel:
        df_train = pd.read_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_train_selected_balanced_multilabel.csv')
        df_test = pd.read_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_test_selected_multilabel.csv')
        label_cols = LABEL_COLS
    else:
        df_train = pd.read_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_train_selected_balanced.csv')
        df_test = pd.read_csv(f'data_processed_{hours_after_trach}hrs/{df_name}_test_selected.csv')
        label_cols = ['label']

    if df_name == 'dyn_full':
        groups_train = df_train['stay_id'].values
        groups_test = df_test['stay_id'].values
        x_df_train = df_train.drop(columns=['stay_id'] + label_cols)
        x_df_test = df_test.drop(columns=['stay_id'] + label_cols)
    else:
        groups_train = None
        groups_test = None
        x_df_train = df_train.drop(columns=label_cols)
        x_df_test = df_test.drop(columns=label_cols)

    y_df_train = df_train[label_cols]
    y_df_test = df_test[label_cols]
    return x_df_train, x_df_test, y_df_train, y_df_test, groups_train, groups_test, label_cols


def _plot_and_save(shap_values, labels, hours_after_trach, model_name, df_name, suffix):
    shap_values.feature_names = labels
    shap.plots.beeswarm(shap_values, max_display=11)
    plt.savefig(
        f'plots_{hours_after_trach}hrs/{model_name}_{df_name}_shap_{suffix}.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


def plot_shap(model_name, df_name, hours_after_trach, multilabel):
    if multilabel:
        model = joblib.load(f'models_{hours_after_trach}hrs/{model_name}_{df_name}_multilabel.pkl')
    else:
        model = joblib.load(f'models_{hours_after_trach}hrs/{model_name}_{df_name}.pkl')

    model_feats = _get_model_features(model)
    labels = _get_feature_labels(model_feats)

    x_df_train, x_df_test, y_df_train, y_df_test, groups_train, groups_test, label_cols = _load_data(
        df_name,
        hours_after_trach,
        multilabel
    )

    x_df_train = x_df_train[model_feats]
    x_df_test = x_df_test[model_feats]

    if multilabel:
        for label_idx, label_name in enumerate(label_cols):
            predict_fn = lambda X, idx=label_idx: model.estimators_[idx].predict_proba(X)[:, 1]
            explainer = shap.Explainer(predict_fn, x_df_train)

            for split_name, x_df, groups in [
                ('train', x_df_train, groups_train),
                ('test', x_df_test, groups_test)
            ]:
                shap_values = explainer(x_df)
                if df_name == 'dyn_full' and groups is not None:
                    shap_values = _aggregate_by_group(shap_values, x_df, groups)
                    split_name = f'{split_name}_by_stay'
                _plot_and_save(
                    shap_values,
                    labels,
                    hours_after_trach,
                    model_name,
                    df_name,
                    f'multilabel_{label_name}_{split_name}'
                )
    else:
        predict_fn = lambda X: model.predict_proba(X)[:, 1]
        explainer = shap.Explainer(predict_fn, x_df_train)
        shap_values = explainer(x_df_test)
        if df_name == 'dyn_full' and groups_test is not None:
            shap_values = _aggregate_by_group(shap_values, x_df_test, groups_test)
            suffix = 'by_stay'
        else:
            suffix = 'test'
        _plot_and_save(shap_values, labels, hours_after_trach, model_name, df_name, suffix)
