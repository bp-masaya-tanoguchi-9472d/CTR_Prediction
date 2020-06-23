import pandas as pd
import numpy as np

import pickle

from sklearn.model_selection import ParameterGrid
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Train Valid
train = pd.read_csv('train.csv',encoding='cp932')
id_date_max = train[['delivery_id','user_id','date']].groupby('user_id').max()['delivery_id']
Train = train.set_index('delivery_id').drop(id_date_max,axis=0)
Valid = train.set_index('delivery_id').loc[id_date_max]
# Test
Test = pd.read_csv('test.csv',encoding='cp932')

def get_data(df):
    df_ = df.drop(['date', 'user_id'],axis=1)

    gender_dummis = pd.get_dummies(df_['gender'],prefix='gender',drop_first=True)
    age_dummies = pd.get_dummies(df_['age'],prefix='gender')
    #age_dummies = df_['age'].replace({'20代':2,'40代':4,'30代':3,'20歳未満':1,'50代以上':5})
    pre_dummies = pd.get_dummies(df_['prefectures'],prefix='prec')
    
    df_2 = df_.drop(['gender', 'age', 'prefectures'],axis=1)
    df_2['gender'] = gender_dummis
    #df_2['age'] = age_dummies
    df_2 = pd.concat([df_2,age_dummies],axis=1)
    df_2 = pd.concat([df_2,pre_dummies],axis=1)
    
    ratio_total = (df_2['prev_total_click']/df_2['prev_total_cnt']).replace(0,-1).fillna(0)
    ratio_eco = (df_2['prev_economy_click']/df_2['prev_economy_cnt']).replace(0,-1).fillna(0)
    ratio_pol = (df_2['prev_politics_click']/df_2['prev_politics_cnt']).replace(0,-1).fillna(0)
    ratio_soc = (df_2['prev_society_click']/df_2['prev_society_cnt']).replace(0,-1).fillna(0)
    ratio_spo = (df_2['prev_sport_click']/df_2['prev_sport_cnt']).replace(0,-1).fillna(0)
    ratio_ent = (df_2['prev_entertainment_click']/df_2['prev_entertainment_cnt']).replace(0,-1).fillna(0)
    
    df_2['ratio_total'] = ratio_total
    df_2['ratio_eco'] = ratio_eco
    df_2['ratio_pol'] = ratio_pol
    df_2['ratio_soc'] = ratio_soc
    df_2['ratio_spo'] = ratio_spo
    df_2['ratio_ent'] = ratio_ent
    return df_2
def delete_type(df,col):
    df_ = df[(df['type']=='%s'%col)]
    return df_.drop('type',axis=1)
def under_sample(df):
    df0 = df[(df['click_flg']==0)]
    df1 = df[(df['click_flg']==1)]
    df0_ = df0.sample(len(df1))
    return pd.concat([df0_,df1])##アンダーサンプルしていない


Train2 = get_data(Train)
Valid2 = get_data(Valid)
#Train 最終データセット
tr_eco = delete_type(Train2,col='経済')
tr_ent = delete_type(Train2,col='エンタメ')
tr_spo = delete_type(Train2,col='スポーツ')
tr_pol = delete_type(Train2,col='政治')
tr_soc = delete_type(Train2,col='社会')

tr_eco_u = under_sample(tr_eco)
tr_ent_u = under_sample(tr_ent)
tr_spo_u = under_sample(tr_spo)
tr_pol_u = under_sample(tr_pol)
tr_soc_u = under_sample(tr_soc)


#Valid　最終データセット
va_eco = delete_type(Valid2,col='経済')
va_ent = delete_type(Valid2,col='エンタメ')
va_spo = delete_type(Valid2,col='スポーツ')
va_pol = delete_type(Valid2,col='政治')
va_soc = delete_type(Valid2,col='社会')

va_eco_u = under_sample(va_eco)
va_ent_u = under_sample(va_ent)
va_spo_u = under_sample(va_spo)
va_pol_u = under_sample(va_pol)
va_soc_u = under_sample(va_soc)


#各typeごとの学習
cols = ['eco','ent','spo','pol','soc']
Trains = [tr_eco_u, tr_ent_u, tr_spo_u, tr_pol_u, tr_soc_u]
Vals = [va_eco_u, va_ent_u, va_spo_u, va_pol_u, va_soc_u]


param_grid = {
        'penalty' : ['l2', 'l1','elasticnet'],
        'C':[0.01,0.1,1,10,50,100],
        'solver':['saga'],
        'l1_ratio':[0.5]
       }

param_dicts = list(ParameterGrid(param_grid))
for (i,(col,tr,va)) in enumerate(zip(cols,Trains,Vals)):
    #train
    y_tr = tr['click_flg'].values
    x_tr = tr.drop('click_flg',axis=1).values
    #val
    y_va = va['click_flg'].values
    x_va = va.drop('click_flg',axis=1).values
    
    #標準化
    ss = StandardScaler()
    ss.fit(x_tr)
    fn_ss = 'ss_%s.sav'%col
    pickle.dump(ss, open(fn_ss, 'wb'))
    
    x_tr_nor = ss.transform(x_tr)#train
    x_va_nor = ss.transform(x_va)#val
    
    #学習
    best_score = 0
    
    for params in param_dicts:
        model = LogisticRegression(**params)
        model.fit(x_tr_nor,y_tr)
        score_tmp = model.score(x_va_nor,y_va)
        if score_tmp > best_score:
            best_params = params
    
    
    
    model = LogisticRegression(**best_params)
    model.fit(x_tr_nor,y_tr)
    pred = model.predict(x_va_nor)
    fn_model = 'model_%s.sav'%col
    pickle.dump(model, open(fn_model, 'wb'))
    
    score = model.score(x_va_nor,y_va)
    
    fpr, tpr, thresholds = metrics.roc_curve(y_va, pred)
    auc = metrics.auc(fpr, tpr)
    
    Score = pd.DataFrame({'type':[col],'score':[auc]})
    if i == 0:
        Scores = Score
    else:
        Scores = pd.concat([Scores,Score])


#標準化　関数
loaded_ss_eco = pickle.load(open('ss_eco.sav', 'rb'))
loaded_ss_ent = pickle.load(open('ss_ent.sav', 'rb'))
loaded_ss_spo = pickle.load(open('ss_spo.sav', 'rb'))
loaded_ss_pol = pickle.load(open('ss_pol.sav', 'rb'))
loaded_ss_soc = pickle.load(open('ss_soc.sav', 'rb'))

#学習モデル
loaded_model_eco = pickle.load(open('model_eco.sav', 'rb'))
loaded_model_ent = pickle.load(open('model_ent.sav', 'rb'))
loaded_model_spo = pickle.load(open('model_spo.sav', 'rb'))
loaded_model_pol = pickle.load(open('model_pol.sav', 'rb'))
loaded_model_soc = pickle.load(open('model_soc.sav', 'rb'))


#テスト予測
Test2 = get_data(Test)

test_eco_nor = loaded_ss_eco.transform(Test2.values)
test_ent_nor = loaded_ss_ent.transform(Test2.values)
test_spo_nor = loaded_ss_spo.transform(Test2.values)
test_pol_nor = loaded_ss_pol.transform(Test2.values)
test_soc_nor = loaded_ss_soc.transform(Test2.values)

proba_eco = loaded_model_eco.predict_proba(test_eco_nor)[:,1]
proba_ent = loaded_model_ent.predict_proba(test_ent_nor)[:,1]
proba_spo = loaded_model_spo.predict_proba(test_spo_nor)[:,1]
proba_pol = loaded_model_pol.predict_proba(test_pol_nor)[:,1]
proba_soc = loaded_model_soc.predict_proba(test_soc_nor)[:,1]

df_probs = pd.DataFrame({'経済':proba_eco,'エンタメ':proba_ent,'スポーツ':proba_spo,'政治':proba_pol,'社会':proba_soc})


Pros = []
Types = []
for idx in df_probs.index:
    pros = df_probs.loc[idx]
    max_val = pros.max()
    pros = df_probs.loc[idx]
    Pros.append(max_val)
    types = pros[(pros==max_val)].index[0]
    Types.append(types)

# submit
submit = Test[['date','user_id']]
submit['type'] = Types
submit['probability'] = Pros
submit.to_csv('answer_test_ver01.tsv', sep = '\t', index=False, encoding='utf-8')