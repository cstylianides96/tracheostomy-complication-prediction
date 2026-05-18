import pandas as pd
from icu_preprocess_util import *
from outlier_removal import *
import os


def create_cohort():
    # Tracheostomy itemids
    itemids  = pd.read_csv('data_raw_mimic2.2/d_items.csv')
    trach_items = itemids[itemids['label'].str.contains('tracheostomy', case=False)]['itemid'].tolist()

    # Cohort undergone tracheostomy
    cohort = pd.read_csv('data_raw_mimic2.2/procedureevents.csv', usecols=['subject_id', 'hadm_id', 'stay_id', 'endtime', 'itemid'])
    cohort = cohort[cohort['itemid'].isin(trach_items)].copy()
    cohort = cohort.rename(columns={'endtime': 'trach_time'})
    cohort['trach_time'] = pd.to_datetime(cohort['trach_time'], errors='coerce')

    # Remove trach_time before icu adm or after discharge
    icustays = pd.read_csv('data_raw_mimic2.2/icustays.csv', usecols=['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'los'])
    cohort = cohort.merge(icustays, on=['subject_id', 'hadm_id', 'stay_id'], how='inner')
    cohort['intime'] = pd.to_datetime(cohort['intime'])
    cohort['outtime'] = pd.to_datetime(cohort['outtime'])
    cohort = cohort[cohort['trach_time']>cohort['intime']]
    cohort = cohort[cohort['trach_time']<cohort['outtime']]
    cohort = cohort.reset_index(drop=True)
    stayids = cohort['stay_id'].tolist()
    
    # Label - Complications
    dx = pd.read_csv('data_raw_mimic2.2/diagnoses_icd.csv',usecols=['subject_id', 'hadm_id', 'icd_code', 'icd_version'])

    # ICD-9 Prefixes for Tracheostomy Complications
    icd9_prefixes = (
        # Bleeding/Hemorrhage
        '9981', '99812', '4590', '7825', '7826', '7847',
        # Infection (post-op infection, acute respiratory failure, bacterial infections, tracheitis, laryngitis)
        '9985', '51901', '041', '5184', '5185', '4818', '4819', '5191',
        # Pneumothorax (all types)
        '5120', '5121', '5122', '5123',
        # Pneumomediastinum
        '5181',
        # Tracheal stenosis
        '5784',
        # Airway obstruction/mechanical complications (includes aspiration, mucus plugging)
        '51902', '51909', '51919', '5189',
        # Tracheo-esophageal fistula
        '53084',
        # Tracheal injury/pressure necrosis
        '51919', '47874',
        # Pressure ulcer/injury from cuff
        '7873', '7874',
        # Innominate artery fistula/vascular injury
        '4434', '9998'
    )

    # ICD-10 Prefixes for Tracheostomy Complications (using prefixes for comprehensive matching)
    icd10_prefixes = [
        # Bleeding/Hemorrhage (post-op hemorrhage, respiratory hemorrhage)
        'T810', 'T811', 'R04', 'R58', 'R060',
        # Infection (post-op infection, acute bronchitis, bronchiolitis, sepsis, tracheitis, mediastinitis)
        'T814', 'J20', 'J21', 'J22', 'A40', 'A41', 'J209', 'J854',
        # Pneumothorax (post-procedural and other pneumothorax)
        'J958', 'J938',
        # Pneumomediastinum
        'J951',
        # Tracheal stenosis (all variants including cuff-related)
        'J395', 'J396',
        # Airway obstruction (includes aspiration, mucus plugging, blood obstruction)
        'J950', 'J951', 'J952', 'J953', 'J9582', 'R0602',
        # Tracheo-esophageal fistula
        'J9504',
        # Tracheal injury/fistula (tracheo-innominate artery fistula)
        'J9501', 'J955',
        # Pressure injury/ulcer from cuff over-inflation
        'L89', 'L974',
        # Innominate artery injury/fistula
        'I710',
        # Other post-procedural complications
        'T815', 'T81'
    ]

    # Super-categories mapping
    # bleeding: hemorrhage and vascular injuries
    # infection: post-op infections, sepsis, respiratory infections
    # mechanical: pneumothorax, stenosis, airway obstruction, fistulas
    # tissue_damage: pressure injuries and necrosis
    # other_postprocedural: general procedural complications
    complication_categories = {
        'bleeding': {
            'icd9': ['9981', '99812', '4590', '7825', '7826', '7847', '4434', '9998'],
            'icd10': ['T810', 'T811', 'R04', 'R58', 'R060', 'I710']
        },
        'infection': {
            'icd9': ['9985', '51901', '041', '5184', '5185', '4818', '4819', '5191'],
            'icd10': ['T814', 'J20', 'J21', 'J22', 'A40', 'A41', 'J209', 'J854']
        },
        'mechanical': {
            'icd9': ['5120', '5121', '5122', '5123', '5181', '5784', '51902', '51909', '51919', '5189', '53084'],
            'icd10': ['J958', 'J938', 'J951', 'J395', 'J396', 'J950', 'J952', 'J953', 'J9582', 'R0602', 'J9504', 'J9501', 'J955']
        },
        # 'tissue_damage': {
        #     'icd9': ['47874', '7873', '7874'],
        #     'icd10': ['L89', 'L974']
        # },
        # 'other_postprocedural': {
        #     'icd9': [],
        #     'icd10': ['T815', 'T81']

        'other': {
            'icd9': ['47874', '7873', '7874'],
            'icd10': ['L89', 'L974', 'T815', 'T81']
        }
    }

    dx['icd_code'] = (dx['icd_code'].astype(str).str.replace('.', '', regex=False).str.upper())

    # Use prefix matching for ICD-9
    dx_icd9 = dx[
        (dx['icd_version'] == 9) &
        (dx['icd_code'].str.startswith(icd9_prefixes))]

    # Use prefix matching for ICD-10
    dx_icd10 = dx[
        (dx['icd_version'] == 10) &
        (dx['icd_code'].apply(lambda x: any(str(x).startswith(prefix) for prefix in icd10_prefixes)))]

    dx_complications = pd.concat([dx_icd9, dx_icd10], ignore_index=True)

    # Labels (ICD codes are assigned on hospital discharge)
    labels = (
        dx_complications[['subject_id', 'hadm_id']] 
        .drop_duplicates()
        .assign(label=1))

    cohort = cohort.merge(
        labels,
        on=['subject_id', 'hadm_id'],
        how='left')

    cohort['label'] = cohort['label'].fillna(0).astype(int)
    cohort = cohort.sort_values('trach_time').drop_duplicates(subset=['stay_id'], keep='last') # keep last trach if multiple
    cohort['time_to_discharge_hours'] = (cohort['outtime'] - cohort['trach_time']).dt.total_seconds() / 3600
    os.makedirs('data_processed', exist_ok=True)
    cohort.to_csv('data_processed/cohort.csv', index=False)

    # Create complication type columns
    cohort_types = cohort.merge(dx_complications.groupby(['subject_id', 'hadm_id']).agg({'icd_code': lambda x: ';'.join(x)}).reset_index(),
                        on=['subject_id', 'hadm_id'], how='left')

    # Map ICD codes to complication categories
    def get_complication_categories(icd_codes_str):
        if pd.isna(icd_codes_str):
            return ''
        
        icd_codes = icd_codes_str.split(';')
        categories = set()
        
        for icd_code in icd_codes:
            for category, versions in complication_categories.items():
                # Check ICD-9 codes
                if any(icd_code.startswith(prefix) for prefix in versions['icd9']):
                    categories.add(category)
                # Check ICD-10 codes
                if any(icd_code.startswith(prefix) for prefix in versions['icd10']):
                    categories.add(category)
        
        return ';'.join(sorted(categories)) if categories else ''

    cohort_types['complication_categories'] = cohort_types['icd_code'].apply(get_complication_categories)

    # One-hot encode the complication_categories column
    for category in complication_categories.keys():
        cohort_types[f'complication_{category}'] = cohort_types['complication_categories'].apply(
            lambda x: 1 if category in str(x).split(';') else 0
        )

    print(cohort_types[['stay_id', 'label', 'complication_categories', 
                        'complication_bleeding', 'complication_infection', 
                        'complication_mechanical',
                        'complication_other']])
    print(cohort.label.value_counts())

    # # Print percentages of positive labels for each complication category
    # print("\nComplication Category Percentages:")
    # for category in complication_categories.keys():
    #     col_name = f'complication_{category}'
    #     percentage = (cohort_types[col_name].sum() / len(cohort_types)) * 100
    #     print(f"{category}: {percentage:.2f}%")

    # keep only patients with complications
    cohort_types = cohort_types[cohort_types['label'] == 1]  

    # Print percentages of positive labels for each complication category
    print("\nComplication Category Percentages:")
    for category in complication_categories.keys():
        col_name = f'complication_{category}'
        percentage = (cohort_types[col_name].sum() / len(cohort_types)) * 100
        print(f"{category}: {percentage:.2f}%")
        
    print(len(cohort_types))
    cohort_types.to_csv('data_processed/cohort_with_complication_types.csv', index=False)
