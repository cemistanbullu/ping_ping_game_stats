import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

df = pd.read_csv("datasets/mesut.txt",header=None,sep=" ")
df.drop(columns=4,inplace=True,axis=1)
df.columns = ["piezzo1","piezzo2","piezzo3","piezzo4"]
# df.head()

def feature(dataframe,sex_h,sex_a,weight_h,weight_a,height_h,height_a,clock,left_hand_h,left_hand_a,age_h,age_a,year_played_h,year_played_a,datatype):
    dataframe["Sex_Home"] = sex_h
    dataframe["Sex_Away"] = sex_a
    dataframe["Weight_Home"] = weight_h
    dataframe["Weight_Away"] = weight_a
    dataframe["Height_Home"] = height_h
    dataframe["Height_Away"] = height_a
    dataframe["Clock"] = clock
    dataframe["Left_hand_Home"] = left_hand_h
    dataframe["Left_hand_Away"] = left_hand_a
    dataframe["Age_Home"] = age_h
    dataframe["Age_Away"] = age_a
    dataframe["Year_Played_Home"] = year_played_h
    dataframe["Year_Played_Away"] = year_played_a
    dataframe["BMI_Home"] = dataframe["Weight_Home"] / ((dataframe["Height_Home"]/100)**2)
    dataframe["BMI_Away"] = dataframe["Weight_Away"] / ((dataframe["Height_Away"]/100)**2)
    dataframe["Total_Piezzo"] = dataframe["piezzo1"] + dataframe["piezzo2"] + dataframe["piezzo3"] + dataframe["piezzo4"]
    dataframe.loc[(dataframe["BMI_Home"] < 20), "BMI_CAT_Home"] = "Thin"
    dataframe.loc[((dataframe["BMI_Home"] >= 20) & (dataframe["BMI_Home"] < 25)), 'BMI_CAT_Home'] = "Normal"
    dataframe.loc[(dataframe["BMI_Home"] >= 25), "BMI_CAT_Home"] = "Fat"
    dataframe.loc[(dataframe["BMI_Away"] < 20), "BMI_CAT_Away"] = "Thin"
    dataframe.loc[((dataframe["BMI_Away"] >= 20) & (dataframe["BMI_Away"] < 25)), 'BMI_CAT_Away'] = "Normal"
    dataframe.loc[(dataframe["BMI_Away"] >= 25), "BMI_CAT_Away"] = "Fat"
    dataframe.loc[(dataframe["Age_Home"] < 18), "AGE_CAT_Home"] = "Child"
    dataframe.loc[((dataframe["Age_Home"] >= 18) & (dataframe["Age_Home"] < 30)), 'AGE_CAT_Home'] = "Young"
    dataframe.loc[(dataframe["Age_Home"] >= 30), "AGE_CAT_Home"] = "Adult"
    dataframe.loc[(dataframe["Age_Away"] < 18), "AGE_CAT_Away"] = "Child"
    dataframe.loc[((dataframe["Age_Away"] >= 18) & (dataframe["Age_Away"] < 30)), 'AGE_CAT_Away'] = "Young"
    dataframe.loc[(dataframe["Age_Away"] >= 30), "AGE_CAT_Away"] = "Adult"
    dataframe.loc[(dataframe["Year_Played_Home"] < 3), "Level_Home"] = "Beginner"
    dataframe.loc[((dataframe["Year_Played_Home"] >= 3) & (dataframe["Year_Played_Home"] < 5)), 'Level_Home'] = "Mid"
    dataframe.loc[(dataframe["Year_Played_Home"] >= 5), "Level_Home"] = "Professional"
    dataframe.loc[(dataframe["Year_Played_Away"] < 3), "Level_Away"] = "Beginner"
    dataframe.loc[((dataframe["Year_Played_Away"] >= 3) & (dataframe["Year_Played_Away"] < 5)), 'Level_Away'] = "Mid"
    dataframe.loc[(dataframe["Year_Played_Away"] >= 5), "Level_Away"] = "Professional"
    dataframe["Data_Type"] = datatype
    return dataframe


df1 = df.loc[:40]
df2 = df.loc[41:81]
df3 = df.loc[82:122]
df4 = df.loc[123:163]
df5 = df.loc[164:204]
df6 = df.loc[205:245]
df7 = df.loc[246:286]
df8 = df.loc[287:327]
df9 = df.loc[328:368]
df10 = df.loc[369:409]

datas = [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10]

listem=[["Female","Female",67, 40,165, 160, 19,"right_hand", "left_hand",20, 22,1,2,"train"],
["Male","Female", 85,53, 190,160, 22, "right_hand","right_hand",21, 21,7,0,"train"],
["Female","Male", 62,69, 178,171, 20, "right_hand","left_hand", 23,19,1,3,"train"],
["Male","Male",83, 78, 185,181, 11, "right_hand","left_hand",21, 22,3,4,"train"],
["Male","Female", 80,59, 187,173, 21, "left_hand","right_hand",22, 23,4,1,"train"],
["Female","Male", 40, 78, 169,178, 9, "right_hand","left_hand",27, 39,5,8,"train"],
["Male","Male", 70,80, 170,190, 10, "left_hand","right_hand",20, 25,2,3,"train"],
["Male","Female",83,90, 176,179, 23, "right_hand","right_hand", 39,25,4,0,"train"],
["Male","Male", 80,83, 187,186, 21, "left_hand","left_hand", 23,22,4,2,"train"],
["Female","Male", 74,92, 172,179, 20, "right_hand","right_hand", 26,24,6,2,"train"]]

df = pd.DataFrame()
for iter,file in enumerate(datas):
    file = file.reset_index(drop=True)
    file = feature(file, *listem[iter])
    df = pd.concat([df,file],ignore_index=True)

# df.to_csv(r'datam.txt' , sep='\t', mode='a')
# df.tail()
# df.info()
# import matplotlib.pyplot as plt
# df["Total_Piezzo"].plot()
# plt.show()
df["Clock"] = df.Clock.apply(lambda x : "morning" if x <= 12 else "evening")
df = df.sample(frac=1).reset_index(drop=True)
def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)
# cat_cols.remove("label")
df = one_hot_encoder(df, cat_cols, drop_first=True)
df.loc[df["Total_Piezzo"] >= 8000, "label"] = np.random.randint(0, 2, df[df["Total_Piezzo"] >= 8000].shape[0])
df.loc[df["Total_Piezzo"] < 8000, "label"] = np.random.randint(0, 1, df[df["Total_Piezzo"] < 8000].shape[0])
df["label"] = df["label"].astype("int")

from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(df[num_cols])
import joblib
from sklearn.ensemble import RandomForestClassifier
y = df["label"]
X = df.drop(["label"], axis=1)
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import *
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
# train hatası
y_pred = rf_model.predict(X_test)
accuracy_score(y_train, y_pred)
precision_score(y_train, y_pred)
recall_score(y_train, y_pred)
f1_score(y_train, y_pred)

print(classification_report(y_test, y_pred))

y_pred = rf_model.predict(X_test)
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
f1_score(y_test, y_pred)

rf_params = {"max_depth": [3,5, 8, None,10],
             "max_features": [3, 5,10,12, 15,18],
             "n_estimators": [100,200, 300,500],
             "min_samples_split": [2, 5, 7,8,10]}

rf_model = RandomForestClassifier(random_state=42)
rf_cv_model = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=1).fit(X_train, y_train)
rf_cv_model.best_params_

rf_tuned = RandomForestClassifier(**rf_cv_model.best_params_).fit(X_train, y_train)

y_pred = rf_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
f1_score(y_test, y_pred)

import pickle
joblib.dump(rf_tuned, "rf.pkl")
pickle.dump(rf_model, open('rf_model5.pkl', 'wb'))

import matplotlib.pyplot as plt
plot_roc_curve(rf_model, X, y)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

import seaborn as sns
from matplotlib import pyplot as plt
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_tuned, X,save=True)




