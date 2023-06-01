import numpy as np 
import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin

class getServscol(BaseEstimator, TransformerMixin):
    def __init__(self, prod=True, sums=True):
        self.prod = prod
        self.sums = sums
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        cols = [X.columns[i] for i in range(11,16)]
        if self.prod: 
            X['servs_TF'] = X[cols].sum(axis=1)>0
        if self.sums: 
            X['servs_count'] = X[cols].sum(axis=1)
        return X
    def get_feature_names_out(self, feature_names_in):
        return feature_names_in

class getIdxcol(BaseEstimator, TransformerMixin):
    def __init__(self, name, getIdx=True):
        self.name = name
        self.getIdx = getIdx
    def fit(self, X, y=None):
        col_idxs = [1,6,9,10] if 'num' in self.name else [2, 3, 4, 5, 7, 8, 11, 12, 13, 14, 15]
        self.cols = [X.columns[i] for i in col_idxs]
        group_mean = X.groupby(self.cols)['cost'].mean().sort_values().reset_index()
        self.concat_df = pd.concat([group_mean, pd.Series(np.arange(len(group_mean)),name=self.name)], axis=1)
        self.concat_df.drop('cost', axis=1, inplace=True)
        return self
    def transform(self, X):
        if self.name not in X.columns and self.getIdx:
            return X.merge(self.concat_df, how='left', on=self.cols)
        return X
    def get_feature_names_out(self, feature_names_in):
        return feature_names_in

class fillIdxcol(BaseEstimator, TransformerMixin):
    def __init__(self, name):
        self.name = name
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if X[self.name].isna().sum()==0:
            return X
        col_idxs = [1,6,9,10] if 'num' in self.name else [2, 3, 4, 5, 7, 8, 11, 12, 13, 14, 15]
        cols = [X.columns[i] for i in col_idxs]
        for i in np.where(X[self.name].isna())[0]:
            empty = X.loc[i, cols+[self.name]]
            get = X[cols+[self.name]].dropna(axis=0)
            get['check'] = (get[cols]==empty[cols]).sum()
            X.at[i,self.name] = get.sort_values(by=['check'], ascending=False).at[0, self.name]
        return X
    def get_feature_names_out(self, feature_names_in):
        return feature_names_in
    
class getRegcol(BaseEstimator, TransformerMixin):
    def __init__(self, name, degree=5, getReg=True):
        self.name = name
        self.degree = degree
        self.getReg = getReg
    def fit(self, X, y=None):
        if self.name[:3]+'_idx' in X.columns:
            col_idxs = [1,6,9,10] if 'num' in self.name else [2, 3, 4, 5, 7, 8, 11, 12, 13, 14, 15]
            self.cols = [X.columns[i] for i in col_idxs]
            group_mean = X.groupby(self.cols)['cost'].mean().sort_values().reset_index()
            x, y = group_mean['cost'].index, group_mean['cost'].values
            self.poly = np.poly1d(np.polyfit(x, y, self.degree))
        return self
    def transform(self, X):
        if self.name[:3]+'_idx' in X.columns and self.getReg:
            X[self.name] = X[self.name[:3]+'_idx'].apply(lambda x: self.poly(x))
        return X
    def get_feature_names_out(self, feature_names_in):
        return feature_names_in

class transCols(BaseEstimator, TransformerMixin):
    def __init__(self, catopt, num_cols, drop_cols):
        self.catopt = catopt
        self.num_cols = num_cols
        self.drop_cols = drop_cols
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if self.catopt:
            cat_cols = []
            for i,idx in enumerate(X.columns):
                if i not in [1,6,9,10] and idx not in ['num_idx','cat_idx','num_reg','cat_reg']+self.num_cols:
                    cat_cols.append(idx)
            X[cat_cols] = X[cat_cols].astype('str')
        for col in self.drop_cols:
            if col in X.columns:
                X.drop(col, axis=1, inplace=True)
        return X
    def get_feature_names_out(self, feature_names_in):
        return feature_names_in
    
    
    
    
    
    
    
# from s3e11_pipeline import getServscol, getIdxcol, fillIdxcol, getRegcol, transcol   
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.compose import make_column_selector
# from sklearn.metrics import make_scorer
# from sklearn.metrics import mean_squared_log_error
# scoring = make_scorer(mean_squared_log_error, squared=False)
# X, y = train.copy(), train['cost'].copy()

# ct_pandas = ColumnTransformer([('ondhot', OneHotEncoder(drop='if_binary', sparse_output=False), make_column_selector(dtype_include=object))
#                        ], remainder='passthrough', verbose_feature_names_out=False).set_output(transform='pandas')
# mypipe = Pipeline([
#                    ('createServs', getServscol(prod=True, sums=True)),
#                    ('createNumIdx', getIdxcol('num_idx', getIdx=True)),
#                    ('createCatIdx', getIdxcol('cat_idx', getIdx=True)),
#                    ('fillNumIdx', fillIdxcol('num_idx')),
#                    ('fillCatIdx', fillIdxcol('cat_idx')),
#                    ('createNumReg', getRegcol('num_reg', degree=5, getReg=True)),
#                    ('createCatReg', getRegcol('cat_reg', degree=5, getReg=True)),
#                    ('changeType_drop', transCols(catopt = False, num_cols=['unit_sales(in millions)','servs_count'], drop_cols=['id','cost'])), # change categories dtype object
#                    ('categories', ct_pandas)])

# from sklearn.ensemble import RandomForestRegressor
# rndFpipe = Pipeline([('preprocessing', mypipe),
#                      ('rndF', RandomForestRegressor())])
# rndFpipe.get_params()






