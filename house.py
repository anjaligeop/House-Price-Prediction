import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import numpy.linalg as npl
import sklearn.ensemble as se
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import datasets, linear_model

from sklearn.preprocessing import StandardScaler 
labelencoder = LabelEncoder()

train=pd.read_csv(r"train.csv")

tx= pd.DataFrame(train,columns=['LotArea','MSZoning','Street', 'Neighborhood', 'HouseStyle','YearBuilt','YearRemodAdd','RoofStyle','Foundation','CentralAir','YrSold','LandContour','Utilities','Condition1','Fireplaces','SaleType','KitchenQual','OverallQual','Heating','OpenPorchSF','GarageType','GarageFinish','GarageCars','GarageArea','GarageQual','FullBath','1stFlrSF'])
X= pd.DataFrame(train,columns=['LotArea','MSZoning', 'Street','Neighborhood', 'HouseStyle','YearBuilt','YearRemodAdd','RoofStyle','Foundation','CentralAir','YrSold','LandContour','Utilities','Condition1','Fireplaces','SaleType','KitchenQual','OverallQual','Heating','OpenPorchSF','GarageType','GarageFinish','GarageCars','GarageArea','GarageQual','FullBath','1stFlrSF'])
street=pd.get_dummies(X['Street'])
mszoning=pd.get_dummies(X['MSZoning'])
house_style=pd.get_dummies(X['HouseStyle'])
garagearea=pd.get_dummies(X['GarageType'])
garagefinish=pd.get_dummies(X['GarageFinish'])
garagequal=pd.get_dummies(X['GarageQual'])
garagetype=pd.get_dummies(X['GarageType'])


areafull=tx['LotArea']+train['GarageArea']
# =============================================================================
X['areafull'] = pd.Series(areafull, index=X.index)
# =============================================================================

X=pd.concat([X,house_style],axis=1)
X=pd.concat([X,street],axis=1)
  
diff_bld_sold=train['YrSold']-train['YearBuilt']






X=pd.concat([X,mszoning],axis=1)
X=pd.concat([X,diff_bld_sold],axis=1)
X=pd.concat([X,garagearea],axis=1)
X=pd.concat([X,garagefinish],axis=1)
X=pd.concat([X,garagequal],axis=1)
       
X=X.iloc[:,0:32].values

X[:,1]= labelencoder.fit_transform(X[:, 1])
# # 
X[:,2]= labelencoder.fit_transform(X[:, 2])

X[:,5]= labelencoder.fit_transform(X[:,5 ])
X[:,6]= labelencoder.fit_transform(X[:,6 ])
X[:,7]= labelencoder.fit_transform(X[:,7 ])

X[:,8]= labelencoder.fit_transform(X[:, 8])
X[:,9]= labelencoder.fit_transform(X[:,9])
X[:,11]= labelencoder.fit_transform(X[:,11])


X[:,10]= labelencoder.fit_transform(X[:, 10])
X[:,12]= labelencoder.fit_transform(X[:,12 ])
X[:,13]= labelencoder.fit_transform(X[:,13 ])
X[:,14]= labelencoder.fit_transform(X[:,14])

X[:,16]= labelencoder.fit_transform(X[:,16 ])


X[:,17]= labelencoder.fit_transform(X[:,17 ])



g=pd.DataFrame(X)

e=pd.DataFrame(X)



e=e.drop(3,axis=1)
e=e.drop(4,axis=1)
e=e.drop(15,axis=1)
e=e.drop(18,axis=1)
e=e.drop(19,axis=1)
e=e.drop(20,axis=1)
e=e.drop(22,axis=1)
e=e.drop(21,axis=1)
e=e.drop(24,axis=1)


X=e.values;
scaler = StandardScaler().fit(X) 
X = scaler.transform(X)
e=pd.DataFrame(X)
# =============================================================================
Y=train['SalePrice']
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.20, random_state=42)



from sklearn.ensemble import GradientBoostingRegressor
gbrt=GradientBoostingRegressor(n_estimators=100,max_depth=2,max_leaf_nodes=20,learning_rate=0.1) 
gbrt.fit(X_train, y_train) 
gbrt.score(X_test,y_test) 
