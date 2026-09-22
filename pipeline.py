from cohort_creation import create_cohort
from data_extraction import extract_all_data
from data_preprocessing import preprocess
from data_preprocessing_multilabel import preprocess_multilabel    
from ml import create_model
from ml_multilabel import create_model_multilabel
from importance import plot_feat_importances
from explainability import plot_shap
from explainability2 import plot_shap2


def run_pipeline(hours_after_trach, model, df_name, multilabel):
    # Step 1: Create Cohort
    create_cohort()
    
    # Step 2: Extract Data
    extract_all_data()

    # Step 3: Preprocess Data
    preprocess(hours_after_trach)
    
    if multilabel:
        # Step 3b: Preprocess Data for Multilabel Classification
        preprocess_multilabel(hours_after_trach)

    # Step 4: Create ML Models for binary Classification
    create_model(model, df_name, hours_after_trach)  

    if multilabel:
        # Step 4b: Create ML Models for Multilabel Classification
        create_model_multilabel(model, df_name, hours_after_trach) 

    # Step 5: Plot Feature Importances
    plot_feat_importances(hours_after_trach, model, df_name, multilabel)

    # Step 6: Plot SHAP values
    plot_shap(model, df_name, hours_after_trach, multilabel)
    plot_shap2(model, df_name, hours_after_trach, multilabel)

run_pipeline(hours_after_trach=12, model='XGB', df_name='agg_flatten_full', multilabel=False)