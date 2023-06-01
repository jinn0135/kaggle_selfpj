import numpy as np 
import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


class CreateCols(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X['g_Id'] = X['PassengerId'].apply(lambda x:x.split('_')[0]).astype(int)
        X['groupSize'] = X['g_Id'].map(lambda x: X['g_Id'].value_counts()[x])
        X['solo'] = X['groupSize']==1
        X['C_deck'] = X['Cabin'].str.split('/').str[0]
        X['C_num'] = X['Cabin'].str.split('/').str[1].astype(float)
        X['C_side'] = X['Cabin'].str.split('/').str[2]
        X['FName'] = X['Name'].str.split(' ').str[1]
        return X
    def get_feature_names_out(self, feature_names_in):
        return feature_names_in

    
class FillAgeWithCat(BaseEstimator, TransformerMixin):
    def __init__(self, func='mean', age_bins=[-1,19,27,38,np.inf]): # func: 'mean', 'median'
        self.func = func
        self.age_bins = age_bins
        self.age_labels = [str(i) for i in range(len(age_bins)-1)]
    def fit(self, X, y=None):
        self.group = X.groupby('groupSize')['Age'].agg(self.func).to_frame()
        return self
    def transform(self, X, y=None):
        for gN in list(X['groupSize'].unique()):
            X.loc[(X['groupSize']==gN)&(X['Age'].isna()), 'Age'] = self.group.loc[gN]['Age']
        X['Age_cat'] = pd.cut(X['Age'], bins=self.age_bins, labels=self.age_labels)
        return X
    def get_feature_names_out(self, feature_names_in):
        return feature_names_in

    
class FillServs(BaseEstimator, TransformerMixin):
    def __init__(self, func='mean'): # func: 'mean', 'median'
        self.func = func
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        for col in ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']:
            X.loc[(X['CryoSleep']==True)&(X[col].isna()), col] = 0
            group = X.groupby('Age_cat')[col].agg(self.func).to_frame()
            for age_cat in list(X['Age_cat'].unique()):
                X.loc[(X['Age_cat']==age_cat)&(X[col].isna()), col] = group.loc[age_cat][col]
        X['total_servs'] = X['RoomService']+X['FoodCourt']+X['ShoppingMall']+X['Spa']+X['VRDeck']
        return X
    def get_feature_names_out(self, feature_names_in):
        return feature_names_in

class FillWithCol(BaseEstimator, TransformerMixin):
    def __init__(self, fill, col=['g_Id']):
        self.fill = fill
        self.col = col
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        total_cols = self.col+[self.fill]
        if self.col==['g_Id']: # g_Id 기준
            df = X[X['groupSize']>1].copy()
        else: # multi_columns 기준
            df = X.copy()
        get = df.groupby(total_cols)[self.fill].size().unstack().fillna(0)
        group = df.groupby(self.col)[self.fill]
        null_index = X[(X[self.fill].isna())&(X[self.col[0]].isin(get.index))].index    
        X.loc[null_index, self.fill] = group.transform(lambda x: x.fillna(pd.Series.mode(x)[0] 
                                                    if not x.isna().values.all() else x))[null_index]
        return X
    def get_feature_names_out(self, feature_names_in):
        return feature_names_in

class FillByFunc(BaseEstimator, TransformerMixin):
    def __init__(self, fill, func='mode'):
        self.fill = fill
        self.func = func
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        for fill_c in self.fill:
            if self.func=='mode': X[fill_c].fillna(X[fill_c].agg(self.func)[0], inplace=True)
            else: X[fill_c].fillna(X[fill_c].agg(self.func), inplace=True)
        return X
    def get_feature_names_out(self, feature_names_in):
        return feature_names_in
    
        
class FillCNUseLin(BaseEstimator, TransformerMixin):
    def __init__(self, fill='C_num'):
        self.fill = fill
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        for deck in list(X['C_deck'].unique()):
            x_data = X.loc[(X[self.fill].notna())&(X['C_deck']==deck),'g_Id']
            y_data = X.loc[(X[self.fill].notna())&(X['C_deck']==deck),self.fill]
            a,c = np.polyfit(x_data, y_data, 1) # ax+by+c = 0 (b=-1)
            null_index = X.loc[(X[self.fill].isna())&(X['C_deck']==deck)].index
            X.loc[null_index, self.fill] = a*X.loc[null_index, 'g_Id'] + c
        return X
    def get_feature_names_out(self, feature_names_in):
        return feature_names_in
    
    
class FillVIP(BaseEstimator, TransformerMixin):
    def __init__(self, p=10):
        self.p = p
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        mid_total = X[X['total_servs']<X['total_servs'].quantile((100-self.p)/100)].groupby('VIP').mean()['total_servs'].mean()
        X.loc[((X['VIP'].isna()) & (X['total_servs']>=mid_total)), 'VIP']=True
        X.loc[((X['VIP'].isna()) & (X['total_servs']<mid_total)), 'VIP']=False
        return X
    def get_feature_names_out(self, feature_names_in):
        return feature_names_in
    
    
class FillCryoSleep(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        cond_1 = (X['total_servs']!=0)&(X['CryoSleep'].isna())
        cond_2 = (X['CryoSleep'].isna()) & (X['C_deck'].isin(['A','F','E','D','T','C']))
        cond_3 = (X['CryoSleep'].isna()) & (X['C_deck'].isin(['G','B']))
        X.loc[cond_1|cond_2, 'CryoSleep'] = False
        X.loc[cond_3, 'CryoSleep'] = True
        return X
    def get_feature_names_out(self, feature_names_in):
        return feature_names_in


class OptionChoose(BaseEstimator, TransformerMixin):
    def __init__(self, age_cat=True, solo=True, totalS=False, serv_cat=False, family=False):
        self.age_cat = age_cat; self.solo = solo
        self.totalS = totalS; self.serv_cat = serv_cat
        self.family = family; self.drop_cols = ['PassengerId','Cabin','Name','g_Id','FName']
        self.dummy_cols = ['solo','Age_cat','HomePlanet','CryoSleep','Destination','VIP','C_deck','C_side']
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if self.age_cat: self.drop_cols.append('Age')
        else: 
            self.drop_cols.append('Age_cat')
            self.dummy_cols.remove('Age_cat')
        if self.solo: self.drop_cols.append('groupSize')
        else: 
            self.drop_cols.append('solo')
            self.dummy_cols.remove('solo')
        if self.totalS: 
            self.drop_cols.extend(['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck'])
            if self.serv_cat:
                serv_bins, serv_labels = [-1,1,725,1467,10000,np.inf], ['0','small','mid','vip','vvip']
                X['Serv_cat'] = pd.cut(X['total_servs'], bins=serv_bins, labels=serv_labels)
                self.dummy_cols.append('Serv_cat')
        else: self.drop_cols.append('total_servs')
        if self.family: self.dummy_cols.append('FName')
        else: self.drop_cols.append('FName')
        X = pd.concat([X, pd.get_dummies(X[self.dummy_cols])], axis=1)
        X.drop(columns=self.drop_cols+self.dummy_cols, inplace=True)
        return X
    def get_feature_names_out(self, feature_names_in):
        return feature_names_in

    
'''
preprocessing = Pipeline([
    ('create_cols', CreateCols()),            # create colms: g_Id, groupSize, solo, C_deck, C_num, C_side, FName
    ('fill_age_withCat', FillAgeWithCat(func='mean', age_bins=[-1,19,27,38,np.inf])),  # fill Age, create Age_cat
    ('fill_serves', FillServs(func='mean')), 
                        # fill services: RoomService, FoodCourt, ShoppingMall, Spa, VRDeck / create total_services
    ('fill_cabinSide1', FillWithCol('C_side',col=['g_Id'])),
    ('fill_cabinSide2', FillWithCol('FName,col=['g_Id']')),
    ('fill_cabinSide3', FillWithCol('C_side',['FName'])),                                        # C_side mode yet
    ('fill_cabinDeck1', FillWithCol('C_deck',col=['g_Id'])),
    ('fill_cabinDeck2', FillWithCol('C_deck',['HomePlanet','Destination','solo'])),              # C_deck mode yet
    ('fill_UseMode', FillByFunc(['C_side','C_deck','Destination'])),
#     ('fill_UseMode', ColumnTransformer([
#         ('impute', SimpleImputer(strategy='most_frequent').set_output(transform='pandas'), ['C_side','C_deck','Destination'])                                                              # ->>>>>> x
#         ], remainder='passthrough').set_output(transform='pandas')),    # fill Cabin_side, Cabin_deck, Destination
    ('fill_cabinNum', FillCNUseLin()),                                                            # fill Cabin_num
    ('fill_homeplanet1', FillWithCol('HomePlanet',col=['g_Id'])),
    ('fill_homeplanet2', FillWithCol('HomePlanet',['C_deck'])),                                  # fill HomePlanet
    ('fill_vip', FillVIP()),                                                                            # fill VIP
    ('fill_cryosleep', FillCryoSleep()),                                                           # fill CryoSleep
    ('options', OptionChoose(age_cat=True, solo=True, totalS=False, serv_cat=False, family=False)),   
                                                     # options choose / create dummy columns(onehot) / drop columns
#     ('categories', OneHotEncoder(categories='auto', drop='if_binary')),                           # ->>>>>>> x
    ('standardScaler', StandardScaler()),                                    # ->>>>>>> standardScaler check please
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False))     # -> PolynomialFeatures
])
'''