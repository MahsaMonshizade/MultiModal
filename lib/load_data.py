# 读取原始数据
import os
import re

import numpy as np
import pandas as pd

def load_data_mean(path,cancer_type):
    path = os.path.join(path,cancer_type)
    meta_data=pd.read_csv(os.path.join(path, "metadata.csv"),index_col=0)
    data_class={'nonIBD':0,'CD':1,'UC':1}
    meta_data['diagnosis']=meta_data['diagnosis'].map(data_class)
    metagenomics = pd.read_csv(os.path.join(path, "metagenomics.csv"), index_col=0)
    metatranscriptomics = pd.read_csv(os.path.join(path, "metatranscriptomics.csv"), index_col=0)
    metabolomics = pd.read_csv(os.path.join(path, "metabolomics.csv"), index_col=0)
    metagenomics_fill = metagenomics.mean(axis=0)
    metatranscriptomics_fill = metatranscriptomics.mean(axis=0)
    metabolomics_fill = metabolomics.mean(axis=0)
    
    external_ids = meta_data.index.values.tolist()
    
    metagenomics_ids = set(metagenomics.index.values)
    new_metagenomics_ids = [value for value in external_ids if value not in metagenomics_ids]
    new_metagenomics_rows = pd.DataFrame({
    'External ID': new_metagenomics_ids,
    **{f: metagenomics_fill[i] for i, f in enumerate(metagenomics.columns[:])}
})
    new_metagenomics_rows.set_index('External ID', inplace=True)
    metagenomics = pd.concat([metagenomics, new_metagenomics_rows])
    
    # print("lol")
    metatranscriptomics_ids = set(metatranscriptomics.index.values)
    new_metatranscriptomics_ids = [value for value in external_ids if value not in metatranscriptomics_ids]
    new_metatranscriptomics_rows = pd.DataFrame({
    'External ID': new_metatranscriptomics_ids,
    **{f: metatranscriptomics_fill[i] for i, f in enumerate(metatranscriptomics.columns[:])}
})
    new_metatranscriptomics_rows.set_index('External ID', inplace=True)
    metatranscriptomics = pd.concat([metatranscriptomics, new_metatranscriptomics_rows])

    metabolomics_ids = set(metabolomics.index.values)
    metabolomics = metabolomics.fillna(metabolomics_fill)
    new_metabolomics_ids = [value for value in external_ids if value not in metabolomics_ids]
    new_metabolomics_rows = pd.DataFrame({
    'External ID': new_metabolomics_ids,
    **{f: metabolomics_fill[i] for i, f in enumerate(metabolomics.columns[:])}
})
    new_metabolomics_rows.set_index('External ID', inplace=True)
    metabolomics = pd.concat([metabolomics, new_metabolomics_rows])

    # Step 5: Create a DataFrame for the new rows
    metagenomics = metagenomics.reindex(external_ids)
    metatranscriptomics = metatranscriptomics.reindex(external_ids)
    metabolomics = metabolomics.reindex(external_ids)

    return [metagenomics, metatranscriptomics, metabolomics, meta_data]

def load_IBD(path, cancer_type, pre_type):
    if pre_type == "mean":
        return load_data_mean(path,cancer_type)
    else:
        print("pre_type error!")