import pandas as pd



cohort = pd.read_csv('data_processed/cohort.csv')
subjectids = cohort['subject_id'].tolist()
hospids = cohort['hadm_id'].tolist()
stayids = cohort['stay_id'].tolist()
trachtimes = cohort['trach_time'].tolist()
admtimes = cohort['intime'].tolist()
distimes = cohort['outtime'].tolist()
itemids = pd.read_csv('data_raw_mimic2.2/d_items.csv')


def extract_chartevents():
    # Chartevents of tracheostomy patients
    chartevents = pd.read_csv('data_raw_mimic2.2/chartevents.csv.gz', compression='gzip', usecols=['stay_id', 'charttime', 'itemid', 'valuenum'])
    print('READ')
    chartevents = chartevents[chartevents['stay_id'].isin(stayids)]

    # Chart items

    vital_items = ['Temperature', 'Respiratory Rate', 'Heart Rate', 'ART BP Systolic', 'ART BP Diastolic',
                'Arterial  O2 Saturation', 'Total PEEP Level', 'EtCO2', 'FIO2 (CH)', 'Spont Vt', 'Paw High']
    lab_items = ['Sodium (serum)', 'Potassium (serum)', 'Calcium non-ionized', 'Phosphorous', 'Magnesium',
                'C Reactive Protein (CRP)', 'WBC', 'Absolute Neutrophil Count', 'Absolute Count - Neuts',
                'Absolute Count - Lymphs', 'Absolute Count - Monos', 'Absolute Count - Basos', 'Fibrinogen', 'INR',
                'Albumin', 'Alkaline Phosphate', 'Total Bilirubin', 'CK (CPK)', 'Creatinine (serum)', 'Glucose (serum)']

    chart_items = vital_items+lab_items
    chart_items = itemids[itemids['label'].isin(chart_items)]['itemid'].tolist()

    chartevents = chartevents[chartevents['itemid'].isin(chart_items)]
    chartevents['charttime'] = pd.to_datetime(chartevents['charttime'])


    # Ensure charttime after trach_time, after icu adm, before discharge
    trach_chart = []
    for stayid, trachtime, admtime, distime in zip(stayids, trachtimes, admtimes, distimes):
        chart = chartevents[
            (chartevents['stay_id'] == stayid) &
            (chartevents['charttime'] > trachtime) &
            (chartevents['charttime'] > admtime) &
            (chartevents['charttime'] < distime)]
        trach_chart.append(chart)
    trach_chart = pd.concat(trach_chart, ignore_index=True)
    print(trach_chart)
    print(f"Number of unique stay_ids: {trach_chart['stay_id'].nunique()}")
    trach_chart.to_csv('data_processed/trach_chartevents.csv', index=False)


def extract_inputevents():
    # Drug Items
    inputevents = pd.read_csv('data_raw_mimic2.2/inputevents.csv', usecols=['stay_id', 'starttime', 'itemid', 'amount'])
    inputevents = inputevents[inputevents['stay_id'].isin(stayids)] 
    print(len(set(inputevents['stay_id']))) #585
    drug_items = ['Vancomycin']
    drug_items = itemids[itemids['label'].isin(drug_items)]['itemid'].tolist()
    inputevents = inputevents[inputevents['itemid'].isin(drug_items)]
    print(len(set(inputevents['stay_id']))) #512 take vancomycin
    inputevents['starttime'] = pd.to_datetime(inputevents['starttime'])

    # Ensure starttime after trach_time, after icu adm, before discharge
    trach_drug = []
    for stayid, trachtime, admtime, distime in zip(stayids, trachtimes, admtimes, distimes):
        drug = inputevents[
            (inputevents['stay_id'] == stayid) &
            (inputevents['starttime'] > trachtime) &
            (inputevents['starttime'] > admtime) &
            (inputevents['starttime'] < distime)]
        trach_drug.append(drug)
    trach_drug = pd.concat(trach_drug, ignore_index=True) # 321 satisfy conditions

    # Add rows for stay_ids not in trach_drug with amount = 0
    missing_stayids = [sid for sid in stayids if sid not in trach_drug['stay_id'].unique()] #264

    missing_rows = pd.DataFrame({
        'stay_id': missing_stayids,
        'starttime': [pd.NA] * len(missing_stayids),
        'itemid': [pd.NA] * len(missing_stayids),
        'amount': [0] * len(missing_stayids)
    })
    trach_drug = pd.concat([trach_drug, missing_rows], ignore_index=True)
    trach_drug.loc[trach_drug['amount'] != 0, 'amount'] = 1
    print(trach_drug.amount.value_counts())
    print(f"Number of unique stay_ids: {trach_drug['stay_id'].nunique()}") #585
    trach_drug.to_csv('data_processed/trach_inputevents.csv', index=False)

# diag = #diagnoses = [Respiratory disease, Hypertension, COPD, Heart DIsease, Diabetes mellitus type II, Malignancy, Stroke, Immunosuppression]
def extract_diagnoses():
    diagnoses = pd.read_csv('data_raw_mimic2.2/diagnoses_icd.csv')
    diagnoses = diagnoses[diagnoses['hadm_id'].isin(hospids)]

    # ICD codes are already stored without dots; normalize to uppercase strings
    diagnoses['icd_code_normalized'] = diagnoses['icd_code'].astype(str).str.upper()

    def code_in_range(code: str, ranges):
        """Check if ICD-9 code (numeric, no dots) falls within any specified ranges."""
        try:
            if str(code).isdigit():
                code_num = int(code)
                for start, end in ranges:
                    if start <= code_num <= end:
                        return True
        except Exception:
            pass
        return False

    def matches_icd10_prefix(code: str, prefixes):
        """Check if ICD-10 code starts with any of the specified prefixes."""
        if pd.isna(code):
            return False
        code_str = str(code).upper()
        return any(code_str.startswith(prefix) for prefix in prefixes)

    # ICD-9 Diagnosis Ranges (codes are stored without dots in diagnoses_icd.csv)
    # Diagnoses: [Respiratory disease, Hypertension, COPD, Heart Disease,
    #             Diabetes mellitus type II, Malignancy, Stroke, Immunosuppression]
    icd9_ranges = [
        (460, 519),        # Respiratory disease: Diseases of respiratory system
        (4010, 40599),     # Hypertension: Essential HTN and hypertensive disease (401.0-405.99)
        (490, 496),        # COPD
        (410, 414),        # Heart disease: Ischemic heart disease
        (420, 429),        # Heart disease: Other forms of heart disease
        (25000, 25093),    # Type 2 diabetes mellitus (250.0-250.93)
        (140, 239),        # Malignancy: Neoplasms (malignant and benign)
        (430, 438),        # Stroke: Cerebrovascular disease
        (2793, 2793),      # Immunosuppression: Unspecified immunity deficiency (279.3)
        (42, 42),          # Immunosuppression: HIV disease (042)
    ]

    # ICD-9 V-codes for immunosuppression (string-based)
    icd9_v_codes = ('V5811',)  # V58.11 immunosuppression (long-term use of medications)

    # ICD-10 Diagnosis Prefixes (prefix matching captures all subcategories)
    icd10_prefixes = [
        'J',                               # Respiratory disease: J00-J99
        'I10', 'I11', 'I12', 'I13', 'I14', 'I15', 'I16',  # Hypertension
        'J44',                             # COPD
        'I20', 'I21', 'I22', 'I23', 'I24', 'I25',        # Ischemic heart disease
        'I50',                             # Heart failure
        'E11',                             # Type 2 diabetes mellitus
        'C',                               # Malignancy: C00-C97
        'D0', 'D1', 'D2', 'D3', 'D4',     # Neoplasms: D00-D49
        'I60', 'I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'I67', 'I68', 'I69',  # Stroke
        'D84',                             # Immunodeficiency disorders
        'D89',                             # Disorders involving immune mechanism
        'Z79',                             # Long-term drug therapy
        'B20'                              # HIV disease
    ]

    # Filter diagnoses based on ICD-9 ranges/V-codes and ICD-10 prefixes
    def is_relevant_diagnosis(code: str) -> bool:
        if matches_icd10_prefix(code, icd10_prefixes):
            return True
        if isinstance(code, str) and code.startswith('V') and code in icd9_v_codes:
            return True
        if code_in_range(code, icd9_ranges):
            return True
        return False

    diagnoses['is_relevant'] = diagnoses['icd_code_normalized'].apply(is_relevant_diagnosis)
    filtered_diags = diagnoses[diagnoses['is_relevant']]
    print(filtered_diags.icd_version.value_counts())
    filtered_diags.to_csv('data_processed/trach_diagnoses.csv', index=False)


def extract_demographics():

    # Age calculation
    age = pd.read_csv("data_raw_mimic2.2/patients.csv")[['subject_id', 'anchor_year', 'anchor_age', 'anchor_year_group', 'gender']]
    age['yob'] = age['anchor_year'] - age['anchor_age']
    age = age.merge(cohort[['subject_id', 'intime']], how='right', on='subject_id')
    age ['intime'] = pd.to_datetime(age['intime'])
    age['age'] = age['intime'].dt.year - age['yob']
    age = age[age['age'] >= 18].reset_index(drop=True)

    # Race
    eth = pd.read_csv("data_raw_mimic2.2/admissions.csv")[['subject_id', 'race']]
    eth = eth.drop_duplicates(subset=['subject_id']) 
    eth = age.merge(eth, how='left', on='subject_id')

    # Extract height and weight from chartevents
    chartevents = pd.read_csv('data_raw_mimic2.2/chartevents.csv.gz', compression='gzip', usecols=['subject_id', 'stay_id','itemid', 'valuenum'])
    chartevents = chartevents[chartevents['stay_id'].isin(stayids)]
    height_weight = ['Height (cm)', 'Admission Weight (Kg)'] 
    height_weight = itemids[itemids['label'].isin(height_weight)]['itemid'].tolist()
    chartevents = chartevents[chartevents['itemid'].isin(height_weight)]
    chartevents = chartevents.pivot_table(index=['subject_id', 'stay_id'], columns='itemid', values='valuenum', aggfunc='first')
    chartevents = chartevents.reset_index()

    # Build demo from cohort to avoid many-to-many duplication
    demo = cohort[['subject_id', 'stay_id']].merge(
        eth[['subject_id', 'age', 'gender', 'race']], how='left', on='subject_id')
    demo = demo.merge(chartevents, how='left', on=['subject_id', 'stay_id'])
    demo = demo[['subject_id', 'stay_id', 'age', 'gender', 'race', height_weight[0], height_weight[1]]]
    print(f"cohort stay_ids: {cohort['stay_id'].nunique()}, demo rows: {len(demo)}, demo unique stay_ids: {demo['stay_id'].nunique()}")
    print(demo.isnull().sum())
    demo = demo.drop_duplicates(subset=['stay_id'])
    demo.to_csv('data_processed/trach_demographics.csv', index=False)

def extract_all_data():
    extract_chartevents()
    extract_inputevents()
    extract_diagnoses()
    extract_demographics()
