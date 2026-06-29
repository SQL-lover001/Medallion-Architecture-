import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = '/Users/smoneysmooth/Desktop/Project Data'
cbsa_char_path = os.path.join(DATA_DIR, 'cbsa-est2024-alldata-char.csv')
cbsa_age_path = os.path.join(DATA_DIR, 'cbsa-est2024-agesex.csv')
csa_char_path = os.path.join(DATA_DIR, 'csa-est2024-alldata-char.csv')

# Connection Credentials
SERVER = 'classdb.ou-mis.com'
DATABASE = '5772S48'
USERNAME = '5772S48'
PASSWORD = 'Password2'

connection_string = f"mssql+pyodbc://{USERNAME}:{PASSWORD}@{SERVER}/{DATABASE}?driver=ODBC+Driver+17+for+SQL+Server"
engine = create_engine(connection_string)

# ============================================================================
# PHASE 1: LOAD BRONZE TABLES (Raw & Untouched - Ingesting ALL 3 Files)
# ============================================================================
print("1. Ingesting ALL 3 raw files into bronze_ tables...")

# File 1: CBSA Character Demographics
cbsa_char_cols = ['SUMLEV', 'CBSA', 'MDIV', 'NAME', 'LSAD', 'YEAR', 'AGEGRP', 'TOT_POP', 'TOT_MALE', 'TOT_FEMALE']
df_cbsa_char_raw = pd.read_csv(cbsa_char_path, encoding="utf-8-sig", usecols=cbsa_char_cols)

# File 2: CBSA Age/Sex Summary Metrics
df_cbsa_age_raw = pd.read_csv(cbsa_age_path, encoding="utf-8-sig")
median_col = 'MEDIAN_AGE_TOT' if 'MEDIAN_AGE_TOT' in df_cbsa_age_raw.columns else df_cbsa_age_raw.columns[-1]
df_cbsa_age_base = df_cbsa_age_raw[['SUMLEV', 'CBSA', 'MDIV', 'NAME', 'LSAD', 'YEAR', 'POPESTIMATE', 'POPEST_MALE', 'POPEST_FEM', median_col]].copy()
df_cbsa_age_base.columns = ['SUMLEV', 'CBSA', 'MDIV', 'NAME', 'LSAD', 'YEAR', 'POPESTIMATE', 'POPEST_MALE', 'POPEST_FEM', 'MEDIAN_AGE_TOT']

# File 3: CSA Combined Statistical Area Character Demographics
csa_cols = ['SUMLEV', 'CSA', 'NAME', 'LSAD', 'YEAR', 'AGEGRP', 'TOT_POP', 'TOT_MALE', 'TOT_FEMALE']
df_csa_char_raw = pd.read_csv(csa_char_path, encoding="utf-8-sig", usecols=csa_cols)

# Use engine.begin() to safely manage transactions and prevent PendingRollbackErrors
with engine.begin() as connection:
    print("Writing to Bronze tables...")
    df_cbsa_char_raw.to_sql('bronze_cbsa_char', con=connection, if_exists='replace', index=False, chunksize=50000)
    df_cbsa_age_base.to_sql('bronze_cbsa_age', con=connection, if_exists='replace', index=False, chunksize=50000)
    df_csa_char_raw.to_sql('bronze_csa_char', con=connection, if_exists='replace', index=False, chunksize=50000)

# ============================================================================
# PHASE 2: TRANSFORM TO SILVER TABLES (Cleaned & Enriched Operational Views)
# ============================================================================
print("2. Transforming data and staging it into silver_ tables...")

# Safely read back from Bronze
with engine.connect() as conn:
    bronze_char = pd.read_sql('SELECT * FROM bronze_cbsa_char WHERE AGEGRP != 0', con=conn)
    bronze_age = pd.read_sql('SELECT * FROM bronze_cbsa_age', con=conn).sort_values(by=['CBSA', 'YEAR'])

# Transform Char data
bronze_char['state_code'] = bronze_char['NAME'].str.strip().str.slice(-2)
silver_char = bronze_char[['CBSA', 'NAME', 'state_code', 'YEAR', 'AGEGRP', 'TOT_POP', 'TOT_MALE', 'TOT_FEMALE']].rename(columns={'NAME': 'metro_name'})

# Transform Metrics data with safe division & rounding
bronze_age['PriorYearPop'] = bronze_age.groupby('CBSA')['POPESTIMATE'].shift(1)
bronze_age['growth_count'] = bronze_age['POPESTIMATE'] - bronze_age['PriorYearPop']
bronze_age['growth_percentage'] = (bronze_age['growth_count'] / bronze_age['PriorYearPop']) * 100
bronze_age = bronze_age.replace([np.inf, -np.inf], np.nan).fillna(0)
bronze_age['growth_percentage'] = bronze_age['growth_percentage'].round(4)

silver_metrics = bronze_age[['CBSA', 'NAME', 'YEAR', 'POPESTIMATE', 'PriorYearPop', 'growth_count', 'growth_percentage', 'MEDIAN_AGE_TOT']].rename(columns={'NAME': 'metro_name', 'MEDIAN_AGE_TOT': 'median_age'})

# Safely commit to Silver
with engine.begin() as connection:
    print("Writing to Silver tables...")
    silver_char.to_sql('silver_clean_demographics', con=connection, if_exists='replace', index=False, chunksize=50000)
    silver_metrics.to_sql('silver_clean_metrics', con=connection, if_exists='replace', index=False, chunksize=50000)

# ============================================================================
# PHASE 3: MODEL TO GOLD TABLES (Optimized Star Schema for Power BI Visuals)
# ============================================================================
print("3. populating and framing out production data star schema inside gold_ tables...")

silver_metrics_df = pd.read_sql('SELECT * FROM silver_clean_metrics', con=engine)
silver_demog_df = pd.read_sql('SELECT * FROM silver_clean_demographics', con=engine)

# Use engine.begin() to open ONE safe transaction block for all Gold tables
with engine.begin() as connection:
    
    # 1. Dimension: Years
    years = pd.DataFrame({'census_year': silver_metrics_df['YEAR'].unique(), 'year_key': silver_metrics_df['YEAR'].unique()})
    years = years.drop_duplicates(subset=['year_key'])
    print("Writing gold_Years...")
    years.to_sql('gold_Years', con=connection, if_exists='append', index=False)

    # 2. Dimension: Metros
    metros = silver_metrics_df[['CBSA', 'metro_name']].drop_duplicates(subset=['CBSA']).copy()
    metros.columns = ['metro_id', 'metro_name']
    metros['metro_key'] = metros['metro_id']
    metros['state_code'] = metros['metro_name'].str.strip().str.slice(-2)
    metros['latitude'] = np.nan
    metros['longitude'] = np.nan
    metros = metros[['metro_key', 'metro_id', 'metro_name', 'state_code', 'latitude', 'longitude']]
    metros = metros.drop_duplicates(subset=['metro_key'])
    print("Writing gold_Metros...")
    metros.to_sql('gold_Metros', con=connection, if_exists='append', index=False)

    # 3. Dimension: AgeGroups
    age_group_mapping = {1: "0-4", 2: "5-9", 3: "10-14", 4: "15-19", 5: "20-24", 6: "25-29", 7: "30-34", 8: "35-39", 9: "40-44", 10: "45-49", 11: "50-54", 12: "55-59", 13: "60-64", 14: "65-69", 15: "70-74", 16: "75-79", 17: "80-84", 18: "85+"}
    age_groups = pd.DataFrame(list(age_group_mapping.items()), columns=['age_group_key', 'age_group'])
    print("Writing gold_AgeGroups...")
    age_groups.to_sql('gold_AgeGroups', con=connection, if_exists='append', index=False)

    # 4. Dimension: Demographics
    demographics = pd.DataFrame([{'demographic_key': 1, 'race_ethnicity': 'Total Population', 'gender': 'Both Genders'}])
    print("Writing gold_Demographics...")
    demographics.to_sql('gold_Demographics', con=connection, if_exists='append', index=False)

   # ============================================================================
    # 5. Fact: Populations (Grouped aggregation to protect multi-state metros)
    # ============================================================================
    pop_fact = silver_demog_df[['YEAR', 'CBSA', 'AGEGRP', 'TOT_POP']].copy()
    pop_fact.columns = ['year_key', 'metro_key', 'age_group_key', 'population_count']
    pop_fact['demographic_key'] = 1
    pop_fact = pop_fact[['year_key', 'metro_key', 'age_group_key', 'demographic_key', 'population_count']]
    
    # INSTEAD OF DROP_DUPLICATES: Group by the keys and SUM the population counts
    pop_fact = pop_fact.groupby(['year_key', 'metro_key', 'age_group_key', 'demographic_key'], as_index=False)['population_count'].sum()
    
    print("Writing gold_Populations in optimized chunks...")
    pop_fact.to_sql('gold_Populations', con=connection, if_exists='append', index=False, chunksize=500)
    
    # Drop records sharing identical composite key values
    pop_fact = pop_fact.drop_duplicates(subset=['year_key', 'metro_key', 'age_group_key', 'demographic_key'])
    print("Writing gold_Populations in optimized chunks...")
    pop_fact.to_sql('gold_Populations', con=connection, if_exists='append', index=False, chunksize=500)

    # 6. Fact: MetroMetrics
    metrics_fact = silver_metrics_df[['YEAR', 'CBSA', 'POPESTIMATE', 'median_age', 'growth_count', 'growth_percentage']].copy()
    metrics_fact.columns = ['year_key', 'metro_key', 'total_population', 'median_age', 'growth_count', 'growth_percentage']
    
    metrics_fact = metrics_fact.drop_duplicates(subset=['year_key', 'metro_key'])
    print("Writing gold_MetroMetrics...")
    metrics_fact.to_sql('gold_MetroMetrics', con=connection, if_exists='append', index=False, chunksize=5000)

print("Pipeline execution complete! All data successfully structured and committed.")

