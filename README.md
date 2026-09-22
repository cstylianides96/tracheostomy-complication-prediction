 # Explainable Complication Status Prediction After Tracheostomy Procedure

 ### Paper
 [accepted]

### Aim

Investigating whether a patient will experience tracheostomy-related complications by the time of hospital discharge (1st outcome; paper) and the type of complication to be experienced (2nd outcome; not published) using 12-hour post-tracheostomy data (MIMIC-IV). The pipeline includes a sensitivity analysis using 24-hour data input and is supported by exlainability (SHAP). 

### System Specifications

Data processing and experiments were conducted on a local workstation running Ubuntu 22.04.5 LTS with Linux kernel 6.8.0. The system was equipped with an Intel Core i9-12900K CPU (16 cores, 24 threads, up to 5.2 GHz) and 62 GB of RAM. Analyses were run on Python >= 3.10. 

---------------------------------------------------
### Steps to use this repository
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
