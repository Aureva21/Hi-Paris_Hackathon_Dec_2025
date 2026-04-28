# Import packages

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load data
X = pd.read_csv('X_train.csv', nrows = 10_000)

# Load glossaire
gloss_ocod = pd.read_excel('Glossaire.xlsx', sheet_name = 'OCOD', header = None, names=['OCOD_ID', 'OCOD_DESC'])
gloss_stratum = pd.read_excel('Glossaire.xlsx', sheet_name = 'STRATUM', header = None, names=['STRATUM_ID', 'STRATUM_DESC'])
gloss_iscedp = pd.read_excel('Glossaire.xlsx', sheet_name = 'ISCEDP', header = None, names=['ISCEDP_ID', 'ISCEDP_DESC'])


# Mapping depuis le glossaire
mapping_iscedp = {
    'lower secondary': 'lower secondary',
    'upper secondary': 'upper secondary',
    'post-secondary': 'post-secondary',
    'short-cycle': 'short-cycle',
    'bachelor': 'bachelor',
    'master': 'master',
    'doctoral': 'doctoral'
}

mapping_ocod = {
    '0': 'army',
    '1': 'manager',
    '2': 'professional',
    '3': 'technician',
    '4': 'admin',
    '5': 'sales',
    '6': 'agriculture',
    '7': 'craftperson',
    '8': 'machinery',
    '9': 'elementary',
    None: 'other'
}

mapping_stratum = {
    'public': 'public',
    'private': 'prive'
}

drop = [
    'Unnamed: 0',
    'CNTRYID',
    'CNTSCHID',
    'CNTSTUID',
    'CYC',
    'NatCen',
    'SUBNATIO',
    'LANGTEST_QQQ',
    'LANGTEST_COG',
    'LANGTEST_PAQ',
    'ST003D02T',
    'ST003D03T',
    'EFFORT2',
    'OCOD1',
    'OCOD2',
    'OCOD3',
    'ISCEDP',
    'STRATUM',
    'CNT',
    'COBN_S',
    'ISCEDP_DESC',
    'STRATUM_DESC'

]

cat_cols = ['OCOD_MOM', 'OCOD_DAD', 'OCOD_SELF', 'ISCEDP_ENCODED', 'STRATUM_ENCODED']


#Dictionnaire de map

stratum_dict = dict(zip(gloss_stratum['STRATUM_ID'], gloss_stratum['STRATUM_DESC']))
iscedp_dict = dict(zip(gloss_iscedp['ISCEDP_ID'], gloss_iscedp['ISCEDP_DESC']))

# Fonctions de mapping
def encode_iscedp_series(s):
    s = s.str.lower()
    out = pd.Series(['other'] * len(s), index=s.index)

    for key, val in mapping_iscedp.items():
        mask = s.str.contains(key, na=False)
        out[mask] = val

    return out

def encode_stratum_series(s):
    s = s.astype(str).str.lower()
    out = pd.Series(['other'] * len(s), index=s.index)

    for key, val in mapping_stratum.items():
        mask = s.str.contains(key, na=False)
        out[mask] = val

    return out

# Fonction pour le df
def transform_features(df):

    df = df.copy()  
    
    df['STRATUM_DESC'] = df['STRATUM'].map(stratum_dict)
    df['ISCEDP_DESC'] = df['ISCEDP'].map(iscedp_dict)

    df['OCOD_MOM'] = df['OCOD1'].astype(str).str[0].map(mapping_ocod).fillna('other')
    df['OCOD_DAD'] = df['OCOD2'].astype(str).str[0].map(mapping_ocod).fillna('other')
    df['OCOD_SELF'] = df['OCOD3'].astype(str).str[0].map(mapping_ocod).fillna('other')
    
    
    df['ISCEDP_ENCODED'] = encode_iscedp_series(df['ISCEDP_DESC'])
    df['STRATUM_ENCODED'] = encode_stratum_series(df['STRATUM_DESC'])
    
    df['SAME_NAT'] = (df['CNT'] == df['COBN_S']).astype(int)
    df['SAME_LANG'] = (df['LANGTEST_COG'] == df['LANGTEST_PAQ']).astype(int)
    
    return df

# Application
X = transform_features(X)

# One hot encoding et scaling

def final_dataset(df):
    df = df.copy()

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=float)
    cat_encoded = ohe.fit_transform(df[cat_cols].astype(str))
    
    df_cat_encoded = pd.DataFrame(cat_encoded, columns=ohe.get_feature_names_out(cat_cols), index=df.index)
    
    df_final = pd.concat([df.drop(columns=cat_cols+drop), df_cat_encoded], axis=1).fillna(0)

    return df_final

X = final_dataset(X)
