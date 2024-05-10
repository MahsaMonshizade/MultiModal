import pandas as pd
from biom import load_table

### create a new metadata 
# metadata_df = pd.read_csv('../Dataset/ibdmdb_data/hmp2_metadata_2018-08-20.csv')
# new_metadata = metadata_df[['Project', 'External ID', 'Participant ID', 'site_sub_coll', 'data_type', 'week_num', 'visit_num', 'interval_days', 'IntervalName', 'site_name', 'Age at diagnosis', 'biopsy_location', 'Rectum cell biopsy:', 'Ileum cell biopsy:', 'diagnosis', 'Antibiotics', 'Chemotherapy', 'BMI', 'race', 'Specify race']]
# new_metadata.to_csv('metadata.csv', index=False)


### hmp2 metabolomics data biom to dataframe
# hmp2_metabolomics_table = load_table('../Dataset/ibdmdb_data/HMP2_metabolomics.biom')
# hmp2_metabolomics_df = hmp2_metabolomics_table.to_dataframe()
# hmp2_metabolomics_df.to_csv('HMP2_metabolomics.csv')

### hmp2 metabolomics data with metadata biom to dataframe
# hmp2_metabolomics_table = load_table('../Dataset/ibdmdb_data/HMP2_metabolomics_w_metadata.biom')
# hmp2_metabolomics_df = hmp2_metabolomics_table.to_dataframe()
# hmp2_metabolomics_df.to_csv('HMP2_metabolomics_w_metadata.csv')
# metadata = hmp2_metabolomics_table.metadata()

### hmp2 metagenomics data taxonomic profile
hmp2_taxonomic_table = load_table('../Dataset/ibdmdb_data/HMP2_Metagenomics_MGX/taxonomic_profiles.biom')
hmp2_taxonomic_df = hmp2_taxonomic_table.to_dataframe()
hmp2_taxonomic_df.to_csv('taxonomic_profiles.csv')

