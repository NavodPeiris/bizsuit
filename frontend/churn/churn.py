import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_auc_score,roc_curve,classification_report,confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from joblib import dump, load
from imblearn.over_sampling import SMOTE

def remove_outlier(col):
    sorted(col)
    Q1,Q3=np.percentile(col,[25,75])
    IQR=Q3-Q1
    lf= Q1-(1.5 * IQR)   # lower fence
    uf= Q3+(1.5 * IQR)   # upper fence
    return lf, uf

def churn_analyze(df):
    input_df = df.copy()
    cus_column = df['CustomerID']
    df.drop(['CustomerID'],axis=1, inplace=True)
    df.drop(['Churn'], axis=1, inplace=True)

    # convert to object type
    df['CityTier'] = df['CityTier'].astype('object')

    cat=[]
    num=[]
    for i in df.columns:
        if df[i].dtype=='object':
            cat.append(i)
        else:
            num.append(i)   

    # null values are set to median
    for i in df.columns:
        if df[i].isnull().sum() > 0:
            df[i].fillna(df[i].median(),inplace=True)

    # fix outliers
    for column in df.columns:
        if df[column].dtype != 'object': 
            lf,uf=remove_outlier(df[column])
            df[column]=np.where(df[column]>uf,uf,df[column])
            df[column]=np.where(df[column]<lf,lf,df[column])

    # add cashback_per_order feature
    df['cashback_per_order'] = df['CashbackAmount'] / df['OrderCount']

    # one hot encode
    df_encoded = pd.get_dummies(df, drop_first=True)

    # scale the values
    scaler = StandardScaler()
    features = df_encoded[num]
    features = scaler.fit_transform(features)
    scaled_df_encoded = df_encoded.copy()
    scaled_df_encoded[num] = features

    # Load the model from the file
    loaded_rf_model = load('churn/RF_model.joblib')

    churn = loaded_rf_model.predict(scaled_df_encoded)
    churn = np.array(churn)

    scaled_df_encoded['churn'] = churn

    # Filter records where churn is True
    churn_list = scaled_df_encoded[scaled_df_encoded['churn'] == True]
    # Add 'CustomerID' column as the first column
    churn_list.insert(0, 'CustomerID', cus_column)

    churn_cus_ids = churn_list['CustomerID']

    input_df_with_churn = input_df
    
    # Update the 'churn' column to True for matching customer IDs
    input_df_with_churn['churn'] = input_df_with_churn['CustomerID'].isin(churn_cus_ids)

    # Reorder columns to make 'CustomerID' the first column
    cols = ['CustomerID'] + [col for col in input_df_with_churn.columns if col != 'CustomerID']
    input_df_with_churn = input_df_with_churn[cols]

    return input_df_with_churn


def churn_train(df):
    try:
        df.drop(['CustomerID'],axis=1, inplace=True)

        # handling null values
        for i in df.columns:
            if df[i].isnull().sum() > 0:
                print(i)
                print('the total null values are:', df[i].isnull().sum())
                print('the datatype is', df[i].dtypes)
                print()

        df['Churn'] = df['Churn'].astype('object')
        df['CityTier'] = df['CityTier'].astype('object')

        cat=[]
        num=[]
        for i in df.columns:
            if df[i].dtype=='object':
                cat.append(i)
            else:
                num.append(i)

        # use median to fill null values
        for i in df.columns:
            if df[i].isnull().sum() > 0:
                df[i].fillna(df[i].median(),inplace=True)

        # checking for null values again
        for i in df.columns:
            if df[i].isnull().sum() > 0:
                print(i)
                print('the total null values are:', df[i].isnull().sum())
                print('the datatype is', df[i].dtypes)
                print()
            else:
                print(i, " - no null values")

        # remove outliers
        for column in df.columns:
            if df[column].dtype != 'object': 
                lf,uf=remove_outlier(df[column])
                df[column]=np.where(df[column]>uf,uf,df[column])
                df[column]=np.where(df[column]<lf,lf,df[column])

        # add cashback_per_order feature
        df['cashback_per_order'] = df['CashbackAmount'] / df['OrderCount']

        # one hot encode
        df_encoded = pd.get_dummies(df, drop_first=True)

        # scale the values
        scaler = StandardScaler()
        features = df_encoded[num]
        features = scaler.fit_transform(features)
        scaled_df_encoded = df_encoded.copy()
        scaled_df_encoded[num] = features

        X=scaled_df_encoded.drop(['Churn_1'],axis=1)
        y=scaled_df_encoded['Churn_1']

        # SMOTE oversampling
        print('Before OverSampling, the shape of X: {}'.format(X.shape)) 
        print('Before OverSampling, the shape of y: {} \n'.format(y.shape)) 
        
        print("Before OverSampling, counts of label '1': {}".format(sum(y == 1))) 
        print("Before OverSampling, counts of label '0': {}".format(sum(y == 0)))
 
        sm = SMOTE(random_state=33)
        X_res, y_res = sm.fit_resample(X, y.ravel())

        print('After OverSampling, the shape of X: {}'.format(X_res.shape)) 
        print('After OverSampling, the shape of y: {} \n'.format(y_res.shape)) 
        
        print("After OverSampling, counts of label '1': {}".format(sum(y_res == 1))) 
        print("After OverSampling, counts of label '0': {}".format(sum(y_res == 0)))

        X_res=pd.DataFrame(X_res)
        #Renaming column name of Target variable
        y_res=pd.DataFrame(y_res)
        y_res.columns = ['Churn_1']
        scaled_df_encoded_smote = pd.concat([X_res,y_res], axis=1)

        X = scaled_df_encoded_smote.drop(['Churn_1'],axis=1)
        y = scaled_df_encoded_smote['Churn_1']
        X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.75, random_state=42)

        # random forrest model
        RF_model=RandomForestClassifier(n_estimators=100,random_state=1)
        RF_model.fit(X_train, y_train)

        y_train_predict = RF_model.predict(X_train)
        train_acc_RF =RF_model.score(X_train, y_train)

        y_test_predict = RF_model.predict(X_test)
        test_acc_RF = RF_model.score(X_test, y_test)

        print("Random Forest model")
        print("Train accuracy: ", train_acc_RF)
        print("test accuracy: ", test_acc_RF)

        # saving best model
        dump(RF_model, 'churn/RF_model.joblib')

        return "Successfully Trained Model"
    
    except Exception as err:
        print(f"Error : ${err}")
        return f"Error: ${err}"