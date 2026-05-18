**Project: PRECIOUS - Improving Personalized Medicine via AI-based Precision Tracheostomy**
_______________________________________________________________________________________

The project addresses two outcomes:
- complication status after Tracheostomy
- complication type after Tracheostomy

1. Install the required packages through the requirements.txt file.
2. Create the *data_raw_mimic2.2* directory with raw mimic-IV v2.2. data.
3. Create the following directories: 
    *data_processed*, *data_processed_12hrs*, *data_processed 24hrs*, 
    *models_12hrs*, *models_24hrs*, 
    *plots_12hrs*, *plots_24hrs*, 
    *predictions_12hrs*, *predictions_24hrs*, 
    *results_12hrs*, *results_24hrs*
4. For the full pipeline for both outcomes, run **pipeline.py**.
    Arguments:
    time_after_trach: number of hours after tracheostomy to be used for temporal data (12 or 24)
    model: 'RF' or 'XGB'
    df_name: 'dyn_full', 'agg_full', 'latest_full', 'flat_full', 'agg_latest_full', 'agg_flatten_full' depending on temporal data structure 
    multilabel: True or False depending on whether complication type after tracheostomy is the desired outcome.
