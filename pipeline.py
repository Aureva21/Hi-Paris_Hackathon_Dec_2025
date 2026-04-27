import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
# Import packages
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class OHEncoding(BaseEstimator, TransformerMixin):
    def __init__(self,
                 gloss_ocod=None,
                 gloss_stratum=None,
                 gloss_iscedp=None,
                 drop=None):
        self.gloss_ocod = gloss_ocod
        self.gloss_stratum = gloss_stratum
        self.gloss_iscedp = gloss_iscedp
        self.drop = drop  # ne pas modifier ici

        # mappings fixes (ne pas les modifier dans fit/transform)
        self.mapping_iscedp = {
            'lower secondary': 'lower secondary',
            'upper secondary': 'upper secondary',
            'post-secondary': 'post-secondary',
            'short-cycle': 'short-cycle',
            'bachelor': 'bachelor',
            'master': 'master',
            'doctoral': 'doctoral'
        }

        self.mapping_ocod = {
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

        self.mapping_stratum = {
            'public': 'public',
            'private': 'prive'
        }

    def fit(self, X, y=None):
        X = pd.DataFrame(X)

        # on travaille sur des copies internes
        self.drop_ = list(self.drop) if self.drop is not None else []

        if self.gloss_stratum is not None:
            self.stratum_map_ = dict(
                zip(self.gloss_stratum['STRATUM_ID'], self.gloss_stratum['STRATUM_DESC'])
            )
        else:
            self.stratum_map_ = {}

        if self.gloss_iscedp is not None:
            self.iscedp_map_ = dict(
                zip(self.gloss_iscedp['ISCEDP_ID'], self.gloss_iscedp['ISCEDP_DESC'])
            )
        else:
            self.iscedp_map_ = {}

        # gloss_ocod n'est même pas utilisé dans ton code actuel, donc je le laisse de côté
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()

        # 1) Ajout STRATUM_DESC / ISCEDP_DESC depuis les glossaires
        if self.stratum_map_:
            df['STRATUM_DESC'] = df['STRATUM'].map(self.stratum_map_)
        if self.iscedp_map_:
            df['ISCEDP_DESC'] = df['ISCEDP'].map(self.iscedp_map_)

        # 2) Mapping OCOD parents / self
        df['OCOD_MOM'] = df['OCOD1'].astype(str).str[0].map(self.mapping_ocod).fillna('other')
        df['OCOD_DAD'] = df['OCOD2'].astype(str).str[0].map(self.mapping_ocod).fillna('other')
        df['OCOD_SELF'] = df['OCOD3'].astype(str).str[0].map(self.mapping_ocod).fillna('other')

        # 3) Encodage ISCEDP / STRATUM
        df['ISCEDP_ENCODED'] = self._encode_iscedp_series(df['ISCEDP_DESC'])
        df['STRATUM_ENCODED'] = self._encode_stratum_series(df['STRATUM_DESC'])

        # 4) Features booléennes / num
        df['SAME_NAT'] = (df['CNT'] == df['COBN_S']).astype(float)
        df['SAME_LANG'] = (df['LANGTEST_COG'] == df['LANGTEST_PAQ']).astype(float)

        # 5) Drop des colonnes brutes
        if self.drop_:
            df = df.drop(columns=[c for c in self.drop_ if c in df.columns])

        return df

    def _encode_iscedp_series(self, s):
        s = s.astype(str).str.lower()
        out = pd.Series('other', index=s.index)
        for key, val in self.mapping_iscedp.items():
            mask = s.str.contains(key, na=False)
            out[mask] = val
        return out

    def _encode_stratum_series(self, s):
        s = s.astype(str).str.lower()
        out = pd.Series('other', index=s.index)
        for key, val in self.mapping_stratum.items():
            mask = s.str.contains(key, na=False)
            out[mask] = val
        return out


class DropHighMissing(BaseEstimator, TransformerMixin):
    """
    Drop les colonnes avec > threshold de NaN, + une liste manuelle.
    threshold est une fraction (0.6 = 60%).
    """
    def __init__(self, threshold: float = 0.6, manual_drop=None):
        self.threshold = threshold
        self.manual_drop = manual_drop if manual_drop is not None else []

    def fit(self, X, y=None):
        X = pd.DataFrame(X)  # au cas où
        missing_frac = X.isna().mean()
        self.auto_drop_cols_ = missing_frac[missing_frac > self.threshold].index.tolist()
        self.manual_drop_ = list(self.manual_drop)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        cols_to_drop = sorted(set(self.auto_drop_cols_) | set(self.manual_drop_))
        cols_to_drop = [c for c in cols_to_drop if c in X.columns]
        return X.drop(columns=cols_to_drop)


class ColumnImputerByList(BaseEstimator, TransformerMixin):
    """
    Impute les colonnes selon 3 listes :
      - impute_zero : NaN -> 0
      - impute_mean : NaN -> moyenne (calculée sur le train)
      - impute_mode : NaN -> valeur la plus fréquente (calculée sur le train)
    """
    def __init__(self, impute_zero=None, impute_mean=None, impute_mode=None):
        self.impute_zero = impute_zero if impute_zero is not None else []
        self.impute_mean = impute_mean if impute_mean is not None else []
        self.impute_mode = impute_mode if impute_mode is not None else []

    def fit(self, X, y=None):
        X = pd.DataFrame(X)

        # On intersecte avec les colonnes réellement présentes
        self.impute_zero_ = [c for c in self.impute_zero if c in X.columns]
        self.impute_mean_ = [c for c in self.impute_mean if c in X.columns]
        self.impute_mode_ = [c for c in self.impute_mode if c in X.columns]

        # Moyennes
        self.mean_values_ = {col: X[col].mean() for col in self.impute_mean_}

        # Modes
        self.mode_values_ = {}
        for col in self.impute_mode_:
            modes = X[col].mode(dropna=True)
            self.mode_values_[col] = modes.iloc[0] if not modes.empty else None

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        # 0
        for col in self.impute_zero_:
            if col in X.columns:
                X[col] = X[col].fillna(0)

        # mean
        for col, m in self.mean_values_.items():
            if col in X.columns:
                X[col] = X[col].fillna(m)

        # mode
        for col, m in self.mode_values_.items():
            if col in X.columns and m is not None:
                X[col] = X[col].fillna(m)

        return X


class YesNoOtherEncoder(BaseEstimator, TransformerMixin):
    """
    Transforme des colonnes yes/no/NaN en 3 colonnes:
      col_yes, col_no, col_other

    Hypothèse actuelle : 1 = yes, 0 = no, NaN = other.
    (à adapter si tes valeurs sont 'Yes'/'No' ou autre)
    """
    def __init__(self, columns=None, drop_original=True):
        self.columns = columns if columns is not None else []
        self.drop_original = drop_original

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        # On garde seulement les colonnes qui existent vraiment
        self.columns_ = [c for c in self.columns if c in X.columns]
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        for col in self.columns_:
            if col not in X.columns:
                continue

            col_yes = f"{col}_yes"
            col_no = f"{col}_no"
            col_other = f"{col}_other"

            # Ici on suppose 0 / 1 / NaN
            X[col_yes] = (X[col] == 1).astype(int)
            X[col_no] = (X[col] == 0).astype(int)
            X[col_other] = X[col].isna().astype(int)

            if self.drop_original:
                X = X.drop(columns=[col])

        return X



# Custom transformer to remove math columns except math_q1_total_timing
class RemoveMathColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Find all columns that contain 'math' (case insensitive)
        math_cols = [col for col in X.columns if 'math' in col.lower()]
        # Keep only math_q1_total_timing and MathScore (if exists)
        cols_to_drop = [col for col in math_cols if col not in ['math_q1_total_timing', 'MathScore']]
        return X.drop(columns=cols_to_drop)

# Custom transformer to process time columns
class ProcessTimeColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Find all columns that contain 'time' or 'timing' (case insensitive)
        time_cols = [col for col in X.columns if 'timing' in col.lower()]
        
        for col in time_cols:
            # Transform: 1/log(time) if time is present, 0 if NaN
            X[col] = X[col].apply(lambda x: 1 / np.log(x) if pd.notna(x) and x > 1 else 0)
        
        return X

# Custom transformer to fill NaN reading and science values with 0
class FillReadingScienceNaN(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Find all columns that contain 'reading' or 'science' (case insensitive)
        reading_science_cols = [col for col in X.columns 
                                if 'reading' in col.lower() or 'science' in col.lower()]
        
        for col in reading_science_cols:
            X[col] = X[col].fillna(0)
        
        return X
