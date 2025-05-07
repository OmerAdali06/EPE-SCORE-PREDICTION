#Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet,ElasticNetCV

#Reading Dataset
data = pd.read_csv("Data.csv")
#Data Preprocessing
data["2_note_mean"] = data["2_note_mean"].fillna(data.groupby("Sex")["2_note_mean"].transform("mean"))
#Encoding Categorical Variables
dms = pd.get_dummies(data[["Sex"]]).astype("float64")
data = data.drop(["Sex"], axis=1).astype("float64")
data = pd.concat([data, dms[["Sex_Female"]]], axis=1)
X = data[["1_note_mean","2_note_mean","Sex_Female"]]
y = data["EPE"]
#Identifying Outlier Scored Observation Units
clf = LocalOutlierFactor(n_neighbors = 10,contamination=0.08)
clf.fit_predict(data)
data_scores = clf.negative_outlier_factor_
np.sort(data_scores)
#Surpessing Outlier Scored Observation Units
X_numeric = data.select_dtypes(include=[np.number])
data["LOF_score"] = clf.fit_predict(X_numeric)
def winsorize_outliers(data,lof_label="LOF_score"):
    data_copy = data.copy()
    for col in data.select_dtypes(include=[np.number]).columns:
        if col == lof_label:
            continue
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3-Q1
        lower_bound = Q1 - (1.5*IQR)
        upper_bound = Q3 + (1.5*IQR)
        data_copy.loc[data[lof_label] == -1,col] = data.loc[data[lof_label] == -1, col].clip(lower_bound,upper_bound)
    return data_copy
data_surpessed = winsorize_outliers(data,lof_label = "LOF_score")
data_surpessed_copy = data_surpessed.copy()
#Model Training
X = data_surpessed_copy[["1_note_mean","2_note_mean","Sex_Female"]]
y = data_surpessed_copy["EPE"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=1)
enet_model = ElasticNet(alpha=1.0, l1_ratio=0.5)
enet_model.fit(X_train,y_train)
y_pred=enet_model.predict(X_test)
#Model Evaluation of Non-Tuned Model
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)
#Tuning The Model
enet_cv_model=ElasticNetCV(cv=10,random_state = 0).fit(X_train,y_train)
y_pred = enet_cv_model.predict(X_test)
#Model Evaluation of Tuned Model
rmse_ = np.sqrt(mean_squared_error(y_test,y_pred))
r2_ = r2_score(y_test,y_pred)

#Developing Equation
enet_cv_model.coef_
enet_cv_model.intercept_
def EPEScorePrediction(x): #Regression Equation: EPE = 33.89 + 0.556 * X
   return 33.89125357785172 + 0.55654177*x