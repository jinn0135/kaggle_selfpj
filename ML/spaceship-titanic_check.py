# from check import 

import numpy as np 
import pandas as pd 

train = pd.read_csv('./spaceship-titanic/train.csv')
test = pd.read_csv('./spaceship-titanic/test.csv')
submission = pd.read_csv('./spaceship-titanic/sample_submission.csv')
train.shape, test.shape, submission.shape # check shape

# Create check columns
check = pd.concat([train, test], axis=0)
check['g_Id'] = check['PassengerId'].apply(lambda x:x.split('_')[0]).astype(int)
check['groupSize'] = check['g_Id'].map(lambda x: check['g_Id'].value_counts()[x])
check['solo'] = check['groupSize']==1
check['C_deck'] = check['Cabin'].str.split('/').str[0]
check['C_num'] = check['Cabin'].str.split('/').str[1].astype(float)
check['C_side'] = check['Cabin'].str.split('/').str[2]

def FillAgeUseGS(X, col, agg):
    group_AgeMean = X.groupby('groupSize')[col].agg(agg).to_frame()
    for gN in list(X['groupSize'].unique()):
        X.loc[(X['groupSize']==gN)&(X[col].isna()), col] = group_AgeMean.loc[gN][col]

# Fill Age
print('Age',check['Age'].isna().sum())
FillAgeUseGS(check, 'Age', 'mean')
print('Age',check['Age'].isna().sum())
print('-'*20,'Age complete','-'*20)

# Create Age_cat
Age_bins, Age_labels = [-1,19,27,38,np.inf], ['child','young adult','adult','elder']
check['Age_cat'] = pd.cut(check['Age'], bins=Age_bins, labels=Age_labels)

def FillServUseAC(X, col, agg):
    X.loc[(X['CryoSleep']==True)&(X[col].isna()), col] = 0
    Age_servMean = X.groupby('Age_cat')[col].agg(agg).to_frame()
    for age_cat in list(X['Age_cat'].unique()):
        X.loc[(X['Age_cat']==age_cat)&(X[col].isna()), col] = Age_servMean.loc[age_cat][col]

# Fill Services
services_li = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
print(check[services_li].isna().sum())
for col in services_li:
    FillServUseAC(check, col, 'mean')
print(check[services_li].isna().sum())
print('-'*20,'Services complete','-'*20)

# Create Family_Name
check['FName'] = check['Name'].str.split(' ').str[1]

def FillWithCol(X, fill, col=['g_Id']):
    total_cols = col+[fill]
    
    if col==['g_Id']: # g_Id 기준
        df = X[X['groupSize']>1].copy()
    else: # multi_columns 기준
        df = X.copy()
    get = X.groupby(total_cols)[fill].size().unstack().fillna(0)
    group = X.groupby(col)[fill]
    null_index = X[(X[fill].isna())&(X[col[0]].isin(get.index))].index    
    X.loc[null_index, fill] = group.transform(lambda x: x.fillna(pd.Series.mode(x)[0] 
                                                if not x.isna().values.all() else x))[null_index]

    
def FillWithCol_ori(X, col, fill, getsolo=False):
    if getsolo:
        null_index = X.loc[X[fill].isna(), fill].index
        group = X.groupby(col)[fill]
    else:
        null_index = X.loc[(X['groupSize']>1)&(X[fill].isna()), fill].index
        group = X[X['groupSize']>1].groupby(['g_Id'])[fill]
    X.loc[X[fill].isna(), fill] = group.transform(lambda x: x.fillna(pd.Series.mode(x)[0]))[null_index]

# Fill Cabin_Side    
cabin_li = ['C_deck','C_num','C_side']
print(check[cabin_li].isna().sum())
for fill in cabin_li:
    FillWithCol(check, fill)
check[cabin_li].isna().sum()   
    
print('FName', check['FName'].isna().sum())
FillWithCol(check, 'FName')
print('FName', check['FName'].isna().sum()) 

FillWithCol(check, 'C_side', ['FName'])
print('C_side', check['C_side'].isna().sum())

def FillFunc(X, fill, func='mode'):
    if func == 'mode':
        X[fill].fillna(X[fill].agg(func)[0], inplace=True)
    else:
        X[fill].fillna(X[fill].agg(func), inplace=True)
        
FillFunc(check, 'C_side', 'mode')
print('C_side', check['C_side'].isna().sum())
print('-'*20,'Cabin_side complete','-'*20)

# Fill Cabin_Deck
FillWithCol(check, 'C_deck', ['HomePlanet','Destination','solo'])
print('C_deck', check['C_deck'].isna().sum())

FillFunc(check, 'C_deck', 'mode')
print('C_deck', check['C_deck'].isna().sum())
print('-'*20,'Cabin_deck complete','-'*20)

def FillCNUseLin(X, fill='C_num'):
    deck_li = sorted(list(X['C_deck'].unique()))
    for deck in deck_li:
        x_data = X.loc[(X[fill].notna())&(X['C_deck']==deck),'g_Id']
        y_data = X.loc[(X[fill].notna())&(X['C_deck']==deck),fill]
        a,c = np.polyfit(x_data, y_data, 1) # ax+by+c = 0 (b=-1)
        null_index = X.loc[(X[fill].isna())&(X['C_deck']==deck)].index
        X.loc[null_index, fill] = a*X.loc[null_index, 'g_Id'] + c

# Fill Cabin_Num
FillCNUseLin(check, 'C_num')
print('C_num', check['C_num'].isna().sum())
print('-'*20,'Cabin_num complete','-'*20)

# Fill HomePlanet
print('HomePlanet', check['HomePlanet'].isna().sum())
FillWithCol(check, 'HomePlanet')
print('HomePlanet', check['HomePlanet'].isna().sum())

FillWithCol(check, 'HomePlanet', ['C_deck'])
print('HomePlanet', check['HomePlanet'].isna().sum())
print('-'*20,'HomePlanet complete','-'*20)

# Fill Destination
print('Destination', check['Destination'].isna().sum())
FillFunc(check, 'Destination', 'mode')
print('Destination', check['Destination'].isna().sum())
print('-'*20,'Destination complete','-'*20)


def FillVIP(X, p=10):     # X=dataframe, p=상위 x퍼센트 제거 후 fillvip
    mid_total = X[X['total_servs']<X['total_servs'].quantile((100-p)/100)].groupby('VIP').mean()['total_servs'].mean()
    X.loc[((X['VIP'].isna()) & (X['total_servs']>=mid_total)), 'VIP']=True
    X.loc[((X['VIP'].isna()) & (X['total_servs']<mid_total)), 'VIP']=False

# Create total_servs
check['total_servs'] = check['RoomService']+check['FoodCourt']+check['ShoppingMall']+check['Spa']+check['VRDeck']

# Fill VIP
print('VIP', check['VIP'].isna().sum())
FillVIP(check)
print('VIP', check['VIP'].isna().sum())
print('-'*20,'VIP complete','-'*20)


def FillCryoSleep(X):
    cond_1 = (X['total_servs']!=0)&(X['CryoSleep'].isna())
    cond_2 = (X['CryoSleep'].isna()) & (X['C_deck'].isin(['A','F','E','D','T','C']))
    cond_3 = (X['CryoSleep'].isna()) & (X['C_deck'].isin(['G','B']))
              
    X.loc[cond_1|cond_2, 'CryoSleep'] = False
    X.loc[cond_3, 'CryoSleep'] = True
              
# Fill CryoSleep
print('CryoSleep', check['CryoSleep'].isna().sum())
FillCryoSleep(check)
print('CryoSleep', check['CryoSleep'].isna().sum())
print('-'*20,'CryoSleep complete','-'*20)

print()
print('-'*20,'complete','-'*20)

y_train = train['Transported'].copy()
X_train = check[~pd.isnull(check['Transported'])].drop(columns=['Transported'])
X_test = check[pd.isnull(check['Transported'])].drop(columns=['Transported'])


# Drop columns
drop_cols_li = ['PassengerId','Cabin','Name','g_Id','FName']

def drop_cols(X, drop_cols_li):
    X.drop(columns=drop_cols_li, inplace=True)
drop_cols(X_train, drop_cols_li)
drop_cols(X_test, drop_cols_li)


# Create dummies
dummy_cols_li = ['solo','Age_cat','HomePlanet','CryoSleep','Destination','VIP','C_deck','C_side']

def concat_dummies(X, dummy_cols_li):
    dummies = pd.get_dummies(X[dummy_cols_li])
    X = pd.concat([X, dummies], axis=1)
    return X.drop(columns=dummy_cols_li)
X_train = concat_dummies(X_train, dummy_cols_li)
X_test = concat_dummies(X_test, dummy_cols_li)
