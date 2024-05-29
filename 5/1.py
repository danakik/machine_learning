import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

days2 = pd.read_csv('./5/DaysInHospital_Y2.csv', index_col='MemberID')
m = pd.read_csv('./5/Members.csv', index_col='MemberID')
claims = pd.read_csv('./5/Claims_Y1.csv', index_col='MemberID')

i = pd.notnull(m.AgeAtFirstClaim)
m.loc[i,'AgeAtFirstClaim'] = m.loc[i,'AgeAtFirstClaim'].apply(lambda s: s.split('-')[0] if s!='80+' else '80')
m.loc[i,'AgeAtFirstClaim'] = m.loc[i,'AgeAtFirstClaim'].apply(lambda s: int(s))

m['AgeAtFirstClaim'] = m['AgeAtFirstClaim'].infer_objects().fillna(-1)
m['Sex'] = m['Sex'].infer_objects().fillna('N')
claims['LengthOfStay'] = claims['LengthOfStay'].infer_objects().fillna(0)

claims.CharlsonIndex = claims.CharlsonIndex.map({'0':0, '1-2':1, '3-4':3, '5+':5})

claims.LengthOfStay = claims.LengthOfStay.map({0:0, '1 day':1, '2 days':2, '3 days':3, '4 days':4,\
    '5 days':5, '6 days':6, '1- 2 weeks':10, '2- 4 weeks':21, '4- 8 weeks':42, '26+ weeks':182})

f_LengthOfStay = claims.groupby(['MemberID'])['LengthOfStay'].sum()
f_Charlson = claims.groupby(['MemberID'])['CharlsonIndex'].max()


data = pd.DataFrame()
data['f_Charlson'] = f_Charlson
data['f_LengthOfStay'] = f_LengthOfStay
data['AgeAtFirstClaim'] = m['AgeAtFirstClaim']
data['ClaimsTruncated'] = days2['ClaimsTruncated']
data['DaysInHospital'] = days2['DaysInHospital']

sex_one_hot = pd.get_dummies(m['Sex'], prefix='Sex')
data = pd.concat([data, sex_one_hot], axis=1)

data = data.dropna()
data.head(5)

def calc(data):
    dataTrain, dataTest = train_test_split(data, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit( dataTrain.loc[:, dataTrain.columns != 'DaysInHospital'], dataTrain['DaysInHospital'] )
    predictionProb = model.predict_proba( dataTest.loc[:, dataTest.columns != 'DaysInHospital'] )
    auc = metrics.roc_auc_score(dataTest['DaysInHospital'], predictionProb[:,1])
    return auc

def calcAUC(data):
    dataTrain, dataTest = train_test_split(data, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit( dataTrain.loc[:, dataTrain.columns != 'DaysInHospital'], dataTrain.DaysInHospital )
    predictionProb = model.predict_proba( dataTest.loc[:, dataTest.columns != 'DaysInHospital'] )
    fpr, tpr, _ = metrics.roc_curve(dataTest['DaysInHospital'], predictionProb[:,1])
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.show()
    print( metrics.roc_auc_score(dataTest['DaysInHospital'], predictionProb[:,1]) )


feature_sets = [
    ['f_Charlson', 'f_LengthOfStay', 'AgeAtFirstClaim', 'ClaimsTruncated', 'Sex_F', 'Sex_M'],
    ['f_Charlson', 'f_LengthOfStay', 'AgeAtFirstClaim', 'ClaimsTruncated', 'Sex_F'],
    ['f_Charlson', 'f_LengthOfStay', 'AgeAtFirstClaim', 'ClaimsTruncated', 'Sex_M'],
    ['f_Charlson', 'f_LengthOfStay', 'AgeAtFirstClaim', 'ClaimsTruncated'],
    ['f_Charlson', 'f_LengthOfStay', 'ClaimsTruncated', 'Sex_F', 'Sex_M'],
    ['f_Charlson', 'f_LengthOfStay', 'ClaimsTruncated', 'Sex_F'],
    ['f_Charlson', 'f_LengthOfStay', 'ClaimsTruncated', 'Sex_M'],
    ['f_Charlson', 'f_LengthOfStay', 'ClaimsTruncated'],
    ['AgeAtFirstClaim', 'ClaimsTruncated', 'Sex_F', 'Sex_M'],
    ['AgeAtFirstClaim', 'ClaimsTruncated', 'Sex_F'],
    ['AgeAtFirstClaim', 'ClaimsTruncated', 'Sex_M'],
    ['AgeAtFirstClaim', 'ClaimsTruncated']
]

best_auc = 0
best_features = None
best_data = None

for features in feature_sets:
    data_temp = data[features + ['DaysInHospital']]
    auc = calc(data_temp)
    """ print('Features:', features)
    print('AUC:', auc)
    print() """
    if auc > best_auc:
        best_data = data_temp
        best_auc = auc
        best_features = features

print('Best Features:', best_features)
print('Best AUC:', best_auc)
calcAUC(best_data)
