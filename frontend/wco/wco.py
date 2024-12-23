import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
pd.options.display.float_format = '{:.2f}'.format
import statsmodels 
from statsmodels import api as sm
import os
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from datetime import timedelta
from joblib import dump, load

def train_wco_models(receivables_data, payables_data):

    # ----------------------- AR CODE STARTS HERE ---------------------------

    df = receivables_data.copy()
    df.head()

    ##Convert the required columns to datetime (sometimes when loaded from excel, it isn't loaded as datetime. thus this change)
    df['Posting_Date'] = pd.to_datetime(df['Posting_Date'], format = '%m/%d/%Y')
    df['Due_Date'] = pd.to_datetime(df['Due_Date'], format = '%m/%d/%Y')
    df['Baseline_Date'] = pd.to_datetime(df['Baseline_Date'], format = '%m/%d/%Y')

    # Convert the  column from string to integer
    df['Total Open Amount_USD'] = df['Total Open Amount_USD'].astype(float).round(1)
    df['Payterm'] = df['Payterm'].astype(int)
    df['DUNNLEVEL'] = df['DUNNLEVEL'].astype(int)
    df['Is Open'] = df['Is Open'].astype(int)
    df['Credit_limit'] = df['Credit_limit'].astype(int)

    rslt_df = df.loc[df['Is Open'] == 0]
    rslt_df['ID'] = range(1, len(rslt_df) + 1)

    #converting date on which payment is made into datetime
    rslt_df['Payment_Date'] = pd.to_datetime(rslt_df['Payment_Date'], format = '%m/%d/%Y')

    ##Calculate Aging
    rslt_df['Payment_flag'] = rslt_df['Due_Date'] - rslt_df['Payment_Date']

    #convert aging which is our y variable into integer (continous variable)
    rslt_df['Payment_flag'] = pd.to_numeric(rslt_df['Payment_flag'].dt.days, downcast='integer')

    #converted payment flag into late, early and ontime flag, this is done as its needed to
    #create features and behavorial aspect at customer level
    conditions = [
        (rslt_df['Payment_flag'] == 0 ),
        (rslt_df['Payment_flag'] < 0 ),
        (rslt_df['Payment_flag'] > 0)]
    choices = ['ontime', 'late', 'early']
    rslt_df['payment_flagname'] = np.select(conditions, choices, default='NA')
    print(rslt_df)

    ##creating customer level features
    customer_pivot = rslt_df.groupby(
        ['Customer Number', 'payment_flagname']
    ).agg(
        {
            # Find the min, max, and sum of the duration column
            'Total Open Amount_USD': "sum",
        }
    ).reset_index()


    customer_pivot_others = rslt_df.groupby(
        ['Customer Number']
    ).agg(
        {
            'Invoice ID': "count",
            'Credit_limit': "mean",
            'DUNNLEVEL': "mean"
        }
    )

    # pivot data 
    reshaped_data = customer_pivot.pivot(index='Customer Number', columns='payment_flagname')
    #replace and fill blank with 0
    df_filled = reshaped_data.fillna(0)

    #rename column name
    new_column_names = {'early': 'TA_early','late': 'TA_late',
                                            'ontime': 'TA_ontime'}
    df_filled.rename(columns = new_column_names,inplace = True)
    print(df_filled)

    # Calculate row sums
    row_sums= df_filled.sum(axis=1)

    #take feature into percentages
    df_row_percent = df_filled.div(row_sums, axis=0) * 100

    df_row_percent.columns = ['_'.join(filter(None, col)).strip() for col in df_row_percent.columns]

    merged_df = pd.merge(customer_pivot_others, df_row_percent, on='Customer Number')

    ##Join invoice level features with customer data
    Masterdata_ar = pd.merge(rslt_df, merged_df, on='Customer Number', how='left')

    # Print the master DataFrame
    print(Masterdata_ar)

    # Convert the integer column to string
    Masterdata_ar['Payterm'] = Masterdata_ar['Payterm'].astype(int)

    # Convert the categorical column into dummy variables
    dummy_df= pd.get_dummies(Masterdata_ar['Region'])

    # Concatenate the original DataFrame with the dummy variables
    Masterdata_ar = pd.concat([Masterdata_ar, dummy_df], axis=1)

    # Print the updated DataFrame
    print(Masterdata_ar)

    #feature at invoice level
    Masterdata_ar['diff_PB'] = Masterdata_ar['Posting_Date'] - Masterdata_ar['Baseline_Date']
    # Assuming you have a DataFrame called 'df' and a column called 'days'
    Masterdata_ar['diff_PB'] =  pd.to_numeric(Masterdata_ar['diff_PB'].dt.days, downcast='integer')

    # rename column name 
    Masterdata_ar = Masterdata_ar.rename(columns={'Total Open Amount_USD_TA_early': 'TA_early',
                                            'Total Open Amount_USD_TA_late':'TA_late',
                                            'Total Open Amount_USD_TA_ontime':'TA_ontime'})

    #features selected for model
    Model_columns = Masterdata_ar.loc[:, ['ID','Invoice ID_x', 'Payterm', 'Total Open Amount_USD','Credit_limit_x','Payment_flag','Credit_limit_y',
                            'MIDWEST','NORTHEAST','SOUTHEAST','SOUTHWEST','WEST','diff_PB']]

    Model_columns_v1 = Model_columns.fillna(0)

    # Split the data into features (X) and target variable (y)
    X = Model_columns_v1.drop('Payment_flag', axis=1)
    y = Model_columns_v1['Payment_flag']

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train_wid = X_train.drop(['Invoice ID_x','ID'], axis=1)
    X_test_wid = X_test.drop(['Invoice ID_x','ID'], axis=1)

    # Create a gradient boosting regression model
    model_gb_ar = GradientBoostingRegressor()

    # Fit the model to the training data
    model_gb_ar.fit(X_train_wid, y_train)

    importances = model_gb_ar.feature_importances_

    # Predict on the training and test data
    y_train_pred = model_gb_ar.predict(X_train_wid)
    y_test_pred = model_gb_ar.predict(X_test_wid)

    # Save the model to a file
    dump(model_gb_ar, 'wco/models/ar_model.joblib')

    # Evaluate the model using R-squared and mean squared error
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    print("Training R-squared:", train_r2)
    print("Test R-squared:", test_r2)
    print("Training Mean Squared Error (MSE):", train_mse)
    print("Test Mean Squared Error (MSE):", test_mse)

    train_predictions = pd.DataFrame(y_train_pred)
    test_predictions = pd.DataFrame(y_test_pred)

    # Reset the index of the DataFrame
    X_train_reset = X_train.reset_index()

    # Display the DataFrame with reset index
    print(X_train_reset)

    # Merge columns using concat along axis=1
    merged_df_v2 = pd.concat([X_train_reset, train_predictions], axis=1)

    # Reset the index of the DataFrame
    X_test_reset = X_test.reset_index()

    # Display the DataFrame with reset index
    print(X_test_reset)

    #merge x and y of test data
    merged_df_test = pd.concat([X_test_reset, test_predictions], axis=1)

    # Merge all data set with predicted value
    merged_df_all = pd.concat([merged_df_v2, merged_df_test], axis=0)

    merged_df_all = merged_df_all.rename(columns={0 :'Predicted_value'})

    # Join a single column from df2 to df1 based on the common identifier
    Final_merge = pd.merge(Masterdata_ar, merged_df_all[['ID','Predicted_value']], on='ID')

    # Display the merged DataFrame
    print(Final_merge)

    # Convert the float column to timedelta
    Final_merge['Predict_Timedelta'] = pd.to_timedelta(Final_merge['Predicted_value'], unit='days')

    #Get date from aging days
    Final_merge['Predicted_Date'] =Final_merge['Due_Date'] + Final_merge['Predict_Timedelta']

    # Convert the datetime column to date
    Final_merge['Predicted_Date']= pd.to_datetime(Final_merge['Predicted_Date']).dt.date

    # Convert the 'Date' column from object to date
    Final_merge['Predicted_Date'] = pd.to_datetime(Final_merge['Predicted_Date'])

    # Get the week number
    Final_merge['Predicted_Week'] = Final_merge['Predicted_Date'].dt.isocalendar().week
    Final_merge['ActualPaid_week'] = Final_merge['Payment_Date'].dt.isocalendar().week
    # Display the DataFrame with the week numbers
    print(Final_merge)

    # ----------------------- AR CODE ENDS HERE ---------------------------



    # ----------------------- AP CODE STARTS HERE ---------------------------

    df_ap=payables_data.copy()

    df_Paid = df_ap[df_ap['Invoice Status'] != 'Unpaid']

    # converting date column from object to datetime
    df_Paid['Posting_Date'] = pd.to_datetime(df_Paid['Posting Date'])
    df_Paid['Due_Date'] = pd.to_datetime(df_Paid['Net Due Date (System Calculated Date)'])
    df_Paid['Invoice Date'] = pd.to_datetime(df_Paid['Invoice Date'])
    df_Paid['Payment Date'] = pd.to_datetime(df_Paid['Payment Date'])

    df_Paid['Invoice Amount'] = df_Paid['Invoice Amount'].astype(float).round(1)
    df_Paid['Overdue'] = df_Paid['Overdue'].astype(float).round(1)
    df_Paid['Total Outstanding amount'] = df_Paid['Total Outstanding amount'].astype(float).round(1)
    df_Paid['Late payment fees'] = df_Paid['Late payment fees'].astype(float).round(1)
    df_Paid['Payterm_n'] = df_Paid['Payterm_n'].astype(int)

    # copying into another dataframe
    Result_dp = df_Paid

    # creating ID column
    Result_dp['ID'] = range(1, len(Result_dp) + 1)

    # Calculate Aging
    Result_dp['Payment_flag'] = Result_dp['Due_Date'] - Result_dp['Payment Date']

    # converting days aging to integer
    Result_dp['Payment_flag'] = pd.to_numeric(Result_dp['Payment_flag'].dt.days, downcast='integer')

    # early,late ontime flag, this will be needed when we create customer level features
    conditions = [
        (Result_dp['Payment_flag'] == 0 ),
        (Result_dp['Payment_flag'] < 0 ),
        (Result_dp['Payment_flag'] > 0)]
    choices = ['ontime', 'late', 'early']
    Result_dp['payment_flagname'] = np.select(conditions, choices, default='NA')
    print(Result_dp)

    Result_dp['payment_flagname'].value_counts()

    vendor_pivot = Result_dp.groupby(
        ['Supplier ID', 'payment_flagname']
    ).agg(
        {
            # Find the min, max, and sum 
            'Invoice Amount': "sum",
            'Late payment fees':'mean',
            'Invoice Number': "count",
            'Overdue': "mean"
        }
    ).reset_index()

    vendor_pivot_others = Result_dp.groupby(
        ['Supplier ID']
    ).agg(
        {
            # Find the min, max, and sum 
            'Late payment fees':'mean',
            'Invoice Number': "count",
            'Overdue': "mean"
        }
    ).reset_index()

    # Pivot the DataFrame based on column 'A', keeping columns 'B', 'C' as is
    reshaped_dataAP = vendor_pivot.pivot(index=['Supplier ID'], columns='payment_flagname', values='Invoice Amount')

    # Display the pivoted DataFrame
    print(reshaped_dataAP)

    # fill na with 0
    df_filled_ap = reshaped_dataAP.fillna(0)

    # Calculate row sums
    row_sums_ap= df_filled_ap.sum(axis=1)

    # taking percent split at vendor level, same as done for AR
    df_percent_ap = df_filled_ap.div(row_sums_ap, axis=0) * 100

    df_ap_others = vendor_pivot_others.loc[:, ['Supplier ID','Invoice Number','Late payment fees','Overdue']]

    print(df_ap_others)

    merged_df_ap = pd.merge(df_percent_ap, df_ap_others, on='Supplier ID')

    ##Join invoice level features with customer data
    Masterdata_ap = pd.merge(Result_dp, merged_df_ap, on='Supplier ID', how='left')

    # Print the master DataFrame
    print(Masterdata_ap)

    # Convert the categorical column into dummy variables
    dummy_df_ap= pd.get_dummies(Masterdata_ap[['Spend Category','Vendor_Type']])

    # Concatenate the original DataFrame with the dummy variables
    Masterdata_ap = pd.concat([Masterdata_ap, dummy_df_ap], axis=1)

    # Print the updated DataFrame
    print(Masterdata_ap)

    Masterdata_ap['diff_PB'] = Masterdata_ap['Posting_Date'] - Masterdata_ap['Invoice Date']

    Masterdata_ap['diff_PB'] =  pd.to_numeric(Masterdata_ap['diff_PB'].dt.days, downcast='integer')

    Model_columns_ap = Masterdata_ap.loc[:, ['ID','Invoice Number_x', 'diff_PB', 'Spend Category_Fees',
                                            'Spend Category_Raw Material','Spend Category_Services','Spend Category_Taxes',
                            'Spend Category_Utility', 'Overdue_y','Payterm_n','Vendor_Type_Domestic','Vendor_Type_International', 'Late payment fees_y','Invoice Number_y',
                            'Payment_flag','Invoice Amount']]

    # Select only the numeric columns
    numeric_cols = Model_columns_ap.select_dtypes(include=[np.number])

    corr_matrix = numeric_cols.corr()

    # Find columns with correlation greater than 0.8
    high_corr_columns = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                colname = corr_matrix.columns[i]
                high_corr_columns.add(colname)

    # Drop the columns with high correlation
    Model_columns_ap_new = Model_columns_ap.drop(high_corr_columns, axis=1)

    print(Model_columns_ap_new)

    Model_columns_AP1 = Model_columns_ap_new.fillna(0)

    # Split the data into features (X) and target variable (y)
    X_AP = Model_columns_AP1.drop('Payment_flag', axis=1)
    y_AP = Model_columns_AP1['Payment_flag']

    # Split the data into train and test sets
    X_trainap, X_testap, y_trainap, y_testap = train_test_split(X_AP, y_AP, test_size=0.3, random_state=42)

    X_train_ap = X_trainap.drop(['Invoice Number_x', 'Invoice Number_y', 'ID'], axis=1)
    X_test_ap = X_testap.drop(['Invoice Number_x', 'Invoice Number_y', 'ID'], axis=1)

    # Create a gradient boosting regression model
    model_gb_ap = GradientBoostingRegressor()

    # Fit the model to the training data
    model_gb_ap.fit(X_train_ap, y_trainap)

    importances_ap = model_gb_ap.feature_importances_

    # Predict on the training and test data
    y_train_predap = model_gb_ap.predict(X_train_ap)
    y_test_predap = model_gb_ap.predict(X_test_ap)

    # Save the model to a file
    dump(model_gb_ap, 'wco/models/ap_model.joblib')

    print("Model saved as 'ap_model.joblib'")

    # Evaluate the model using R-squared and mean squared error
    train_r2ap = r2_score(y_trainap, y_train_predap)
    test_r2ap = r2_score(y_testap, y_test_predap)
    train_mseap = mean_squared_error(y_trainap, y_train_predap)
    test_mseap = mean_squared_error(y_testap, y_test_predap)

    print("Training R-squared:", train_r2ap)
    print("Test R-squared:", test_r2ap)
    print("Training Mean Squared Error (MSE):", train_mseap)
    print("Test Mean Squared Error (MSE):", test_mseap)

    # Convert the NumPy array to a DataFrame
    train_predictions_ap = pd.DataFrame(y_train_predap)
    test_predictions_ap = pd.DataFrame(y_test_predap)
    # Display the DataFrame
    print(train_predictions_ap)
    print(test_predictions_ap)

    # Reset the index of the DataFrame
    X_train_reset = X_train_ap.reset_index()

    # Display the DataFrame with reset index
    print(X_train_reset)

    # Merge columns using concat along axis=1
    merged_df_v2_ap = pd.concat([X_train_reset, train_predictions_ap], axis=1)

    # Reset the index of the DataFrame
    X_test_reset_ap = X_test_ap.reset_index()

    # Display the DataFrame with reset index
    print(X_test_reset_ap)

    merged_df_test_ap = pd.concat([X_test_reset_ap, test_predictions_ap], axis=1)
    merged_df_all_ap = pd.concat([merged_df_v2_ap, merged_df_test_ap], axis=0)

    merged_df_all_ap = merged_df_all_ap.rename(columns={0 :'Predicted_value'})
    Masterdata_ap = Masterdata_ap.reset_index()

    # Join a single column from df2 to df1 based on the common identifier
    Final_merge_ap = pd.merge(Masterdata_ap, merged_df_all_ap[['index','Predicted_value']], on='index')

    # Display the merged DataFrame
    print(Final_merge_ap)

    # Convert the float column to timedelta
    Final_merge_ap['Predict_Timedelta'] = pd.to_timedelta(Final_merge_ap['Predicted_value'], unit='days')
    Final_merge_ap['Predicted_Date'] =Final_merge_ap['Due_Date'] + Final_merge_ap['Predict_Timedelta']

    # Convert the datetime column to date
    Final_merge_ap['Predicted_Date']= pd.to_datetime(Final_merge_ap['Predicted_Date']).dt.date
    # Convert the 'Date' column from object to date
    Final_merge_ap['Predicted_Date'] = pd.to_datetime(Final_merge_ap['Predicted_Date'])

    # Get the week number
    Final_merge_ap['Predicted_Week'] = Final_merge_ap['Predicted_Date'].dt.isocalendar().week
    Final_merge_ap['ActualPaid_week'] = Final_merge_ap['Payment Date'].dt.isocalendar().week
    # Display the DataFrame with the week numbers
    print(Final_merge_ap)

    # Select columns 'B' and 'D' using loc accessor
    Week_AR = Final_merge.loc[:, ['Predicted_Week', 'Total Open Amount_USD']]

    Week_AP = Final_merge_ap.loc[:, ['Predicted_Week', 'Invoice Amount']]

    APBYWEEK = Week_AP.groupby(
        ['Predicted_Week']
    ).agg(
        {
            # Find the min, max, and sum of the duration column
            'Invoice Amount':'sum',
        }
    ).reset_index()

    ARBYWEEK = Week_AR.groupby(
        ['Predicted_Week']
    ).agg(
        {
            # Find the min, max, and sum of the duration column
            'Total Open Amount_USD':'sum',
        }
    ).reset_index()

    new_column_names_ar = {'Total Open Amount_USD': 'Amount_AR'}
    ARBYWEEK.rename(columns = new_column_names_ar,inplace = True)
    print(ARBYWEEK)

    new_column_names_ap = {'Invoice Amount': 'Amount_AP'}
    APBYWEEK.rename(columns = new_column_names_ap,inplace = True)
    print(APBYWEEK)

    ARBYWEEK_AMNT = ARBYWEEK.loc[:, ['Amount_AR']]
    APBYWEEK_AMNT = APBYWEEK.loc[:, ['Amount_AP']]

    WCOBYWEEK_1 = pd.merge(ARBYWEEK, APBYWEEK, on='Predicted_Week')

    new_column_names_wco = {'Total Open Amount_USD': 'Amount_AR','Invoice Amount': 'Amount_AP'}
    WCOBYWEEK_1.rename(columns = new_column_names_wco,inplace = True)
    print(WCOBYWEEK_1)

    WCOBYWEEK_1['Working_Capital'] = WCOBYWEEK_1['Amount_AR'] - WCOBYWEEK_1['Amount_AP']

    return "Successfully Trained Models"


def use_wco_models(receivables_data, payables_data):

    # ----------------------prepare receivable data --------------------
    
    ##Convert the required columns to datetime (sometimes when loaded from excel, it isn't loaded as datetime. thus this change)
    receivables_data['Posting_Date'] = pd.to_datetime(receivables_data['Posting_Date'], format = '%m/%d/%Y')
    receivables_data['Due_Date'] = pd.to_datetime(receivables_data['Due_Date'], format = '%m/%d/%Y')
    receivables_data['Baseline_Date'] = pd.to_datetime(receivables_data['Baseline_Date'], format = '%m/%d/%Y')

    # Convert the  column from string to integer
    receivables_data['Total Open Amount_USD'] = receivables_data['Total Open Amount_USD'].astype(float).round(1)
    receivables_data['Payterm'] = receivables_data['Payterm'].astype(int)
    receivables_data['DUNNLEVEL'] = receivables_data['DUNNLEVEL'].astype(int)
    receivables_data['Is Open'] = receivables_data['Is Open'].astype(int)
    receivables_data['Credit_limit'] = receivables_data['Credit_limit'].astype(int)

    rslt_df = receivables_data.loc[receivables_data['Is Open'] == 1]
    rslt_df['ID'] = range(1, len(rslt_df) + 1)

    ##creating customer level features
    customer_pivot = rslt_df.groupby(
        ['Customer Number']
    ).agg(
        {
            # Find the min, max, and sum of the duration column
            'Total Open Amount_USD': "sum",
        }
    ).reset_index()


    customer_pivot_others = rslt_df.groupby(
        ['Customer Number']
    ).agg(
        {
            'Invoice ID': "count",
            'Credit_limit': "mean",
            'DUNNLEVEL': "mean"
        }
    )

    print(customer_pivot)

    merged_df = pd.merge(customer_pivot_others, customer_pivot, on='Customer Number')

    ##Join invoice level features with customer data
    Masterdata_ar = pd.merge(rslt_df, merged_df, on='Customer Number', how='left')

    # Print the master DataFrame
    print(Masterdata_ar)

    # Convert the integer column to string
    Masterdata_ar['Payterm'] = Masterdata_ar['Payterm'].astype(int)

    # Convert the categorical column into dummy variables
    dummy_df= pd.get_dummies(Masterdata_ar['Region'])

    # Concatenate the original DataFrame with the dummy variables
    Masterdata_ar = pd.concat([Masterdata_ar, dummy_df], axis=1)

    # Print the updated DataFrame
    print(Masterdata_ar)

    for col in Masterdata_ar:
        print (col)

    #feature at invoice level
    Masterdata_ar['diff_PB'] = Masterdata_ar['Posting_Date'] - Masterdata_ar['Baseline_Date']
    # Assuming you have a DataFrame called 'df' and a column called 'days'
    Masterdata_ar['diff_PB'] =  pd.to_numeric(Masterdata_ar['diff_PB'].dt.days, downcast='integer')

    Masterdata_ar.rename(columns={'Total Open Amount_USD_x': 'Total Open Amount_USD'}, inplace=True)

    #features selected for model
    Model_columns = Masterdata_ar.loc[:, ['ID','Invoice ID_x', 'Payterm', 'Total Open Amount_USD','Credit_limit_x','Credit_limit_y',
                            'MIDWEST','NORTHEAST','SOUTHEAST','SOUTHWEST','WEST','diff_PB']]

    Model_columns_v1 = Model_columns.fillna(0)

    # Split the data into features (X) and target variable (y)
    # X_AR = Model_columns_v1.drop('Payment_flag', axis=1)
    X_AR = Model_columns_v1.drop(['Invoice ID_x','ID'], axis=1)

    # ----------------------prepare payables data --------------------
    df_Unpaid = payables_data[payables_data['Invoice Status'] == 'Unpaid']

    # converting date column from object to datetime
    df_Unpaid['Posting_Date'] = pd.to_datetime(df_Unpaid['Posting Date'])
    df_Unpaid['Due_Date'] = pd.to_datetime(df_Unpaid['Net Due Date (System Calculated Date)'])
    df_Unpaid['Invoice Date'] = pd.to_datetime(df_Unpaid['Invoice Date'])

    df_Unpaid['Invoice Amount'] = df_Unpaid['Invoice Amount'].astype(float).round(1)
    df_Unpaid['Overdue'] = df_Unpaid['Overdue'].astype(float).round(1)
    df_Unpaid['Total Outstanding amount'] = df_Unpaid['Total Outstanding amount'].astype(float).round(1)
    df_Unpaid['Late payment fees'] = df_Unpaid['Late payment fees'].astype(float).round(1)
    df_Unpaid['Payterm_n'] = df_Unpaid['Payterm_n'].astype(int)

    # copying into another dataframe
    Result_dp = df_Unpaid

    # creating ID column
    Result_dp['ID'] = range(1, len(Result_dp) + 1)

    vendor_pivot = Result_dp.groupby(
        ['Supplier ID']
    ).agg(
        {
            # Find the min, max, and sum 
            'Invoice Amount': "sum",
            'Late payment fees':'mean',
            'Invoice Number': "count",
            'Overdue': "mean"
        }
    ).reset_index()

    vendor_pivot_others = Result_dp.groupby(
        ['Supplier ID']
    ).agg(
        {
            # Find the min, max, and sum 
            'Late payment fees':'mean',
            'Invoice Number': "count",
            'Overdue': "mean"
        }
    ).reset_index()

   
    df_ap_others = vendor_pivot_others.loc[:, ['Supplier ID','Invoice Number','Late payment fees','Overdue']]
    print(df_ap_others)

    merged_df_ap = pd.merge(vendor_pivot, df_ap_others, on='Supplier ID')

    ##Join invoice level features with customer data
    Masterdata_ap = pd.merge(Result_dp, merged_df_ap, on='Supplier ID', how='left')

    # Print the master DataFrame
    print(Masterdata_ap)

    # Convert the categorical column into dummy variables
    dummy_df_ap= pd.get_dummies(Masterdata_ap[['Spend Category','Vendor_Type']])

    # Concatenate the original DataFrame with the dummy variables
    Masterdata_ap = pd.concat([Masterdata_ap, dummy_df_ap], axis=1)

    # Print the updated DataFrame
    print(Masterdata_ap)

    for col in Masterdata_ap:
        print(col)

    Masterdata_ap['diff_PB'] = Masterdata_ap['Posting_Date'] - Masterdata_ap['Invoice Date']

    Masterdata_ap['diff_PB'] =  pd.to_numeric(Masterdata_ap['diff_PB'].dt.days, downcast='integer')

    Masterdata_ap.rename(columns={'Invoice Amount_y': 'Invoice Amount'}, inplace=True)

    Model_columns_ap = Masterdata_ap.loc[:, ['ID','Invoice Number_x', 'diff_PB', 'Spend Category_Fees',
                                            'Spend Category_Raw Material','Spend Category_Services','Spend Category_Taxes',
                            'Spend Category_Utility', 'Overdue_y','Payterm_n','Vendor_Type_Domestic','Vendor_Type_International', 'Late payment fees_y','Invoice Number_y',
                            'Invoice Amount']]

    # Select only the numeric columns
    numeric_cols = Model_columns_ap.select_dtypes(include=[np.number])

    corr_matrix = numeric_cols.corr()

    # Find columns with correlation greater than 0.8
    high_corr_columns = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                colname = corr_matrix.columns[i]
                high_corr_columns.add(colname)

    # Drop the columns with high correlation
    Model_columns_ap_new = Model_columns_ap.drop(high_corr_columns, axis=1)

    print(Model_columns_ap_new)

    Model_columns_AP1 = Model_columns_ap_new.fillna(0)

    # Split the data into features (X) and target variable (y)
    X_AP = Model_columns_AP1.drop(['Invoice Number_x','ID'], axis=1)
    

    # Load the model
    model_gb_ar = load('wco/models/ar_model.joblib')
    model_gb_ap = load('wco/models/ap_model.joblib')

    ar_pred = model_gb_ar.predict(X_AR)
    ap_pred = model_gb_ap.predict(X_AP)

    ar_pred_df = pd.DataFrame(ar_pred)
    ap_pred_df = pd.DataFrame(ap_pred)

    ar_pred_df = ar_pred_df.rename(columns={0 :'Predicted_value'})
    ap_pred_df = ap_pred_df.rename(columns={0 :'Predicted_value'})

    ar_pred_df['ID'] = range(1, len(ar_pred_df) + 1)
    ap_pred_df['ID'] = range(1, len(ap_pred_df) + 1)

    # Join a single column from df2 to df1 based on the common identifier
    Final_merge_ar = pd.merge(Masterdata_ar, ar_pred_df[['ID','Predicted_value']], on='ID')

    # Display the merged DataFrame
    print(Final_merge_ar)

    # Convert the float column to timedelta
    Final_merge_ar['Predict_Timedelta'] = pd.to_timedelta(Final_merge_ar['Predicted_value'], unit='days')

    #Get date from aging days
    Final_merge_ar['Predicted_Date'] = Final_merge_ar['Due_Date'] + Final_merge_ar['Predict_Timedelta']

    # Convert the datetime column to date
    Final_merge_ar['Predicted_Date']= pd.to_datetime(Final_merge_ar['Predicted_Date']).dt.date

    # Convert the 'Date' column from object to date
    Final_merge_ar['Predicted_Date'] = pd.to_datetime(Final_merge_ar['Predicted_Date'])

    # Get the week number
    Final_merge_ar['Predicted_Week'] = Final_merge_ar['Predicted_Date'].dt.isocalendar().week

    # Display the DataFrame with the week numbers
    print(Final_merge_ar)
    

    
    # Join a single column from df2 to df1 based on the common identifier
    Final_merge_ap = pd.merge(Masterdata_ap, ap_pred_df[['ID','Predicted_value']], on='ID')

    # Display the merged DataFrame
    print(Final_merge_ap)

    # Convert the float column to timedelta
    Final_merge_ap['Predict_Timedelta'] = pd.to_timedelta(Final_merge_ap['Predicted_value'], unit='days')
    Final_merge_ap['Predicted_Date'] =Final_merge_ap['Due_Date'] + Final_merge_ap['Predict_Timedelta']

    # Convert the datetime column to date
    Final_merge_ap['Predicted_Date']= pd.to_datetime(Final_merge_ap['Predicted_Date']).dt.date
    # Convert the 'Date' column from object to date
    Final_merge_ap['Predicted_Date'] = pd.to_datetime(Final_merge_ap['Predicted_Date'])

    # Get the week number
    Final_merge_ap['Predicted_Week'] = Final_merge_ap['Predicted_Date'].dt.isocalendar().week
 
    # Display the DataFrame with the week numbers
    print(Final_merge_ap)

    Week_AR = Final_merge_ar.loc[:, ['Predicted_Week', 'Total Open Amount_USD']]

    Week_AP = Final_merge_ap.loc[:, ['Predicted_Week', 'Invoice Amount']]

    APBYWEEK = Week_AP.groupby(
        ['Predicted_Week']
    ).agg(
        {
            # Find the min, max, and sum of the duration column
            'Invoice Amount':'sum',
        }
    ).reset_index()

    ARBYWEEK = Week_AR.groupby(
        ['Predicted_Week']
    ).agg(
        {
            # Find the min, max, and sum of the duration column
            'Total Open Amount_USD':'sum',
        }
    ).reset_index()

    new_column_names_ar = {'Total Open Amount_USD': 'Amount_AR'}
    ARBYWEEK.rename(columns = new_column_names_ar,inplace = True)
    print(ARBYWEEK)

    new_column_names_ap = {'Invoice Amount': 'Amount_AP'}
    APBYWEEK.rename(columns = new_column_names_ap,inplace = True)
    print(APBYWEEK)

    ARBYWEEK_AMNT = ARBYWEEK.loc[:, ['Amount_AR']]
    APBYWEEK_AMNT = APBYWEEK.loc[:, ['Amount_AP']]

    WCOBYWEEK = pd.merge(ARBYWEEK, APBYWEEK, on='Predicted_Week')

    new_column_names_wco = {'Total Open Amount_USD': 'Amount_AR','Invoice Amount': 'Amount_AP'}
    WCOBYWEEK.rename(columns = new_column_names_wco,inplace = True)

    WCOBYWEEK['Working_Capital'] = WCOBYWEEK['Amount_AR'] - WCOBYWEEK['Amount_AP']
    print(WCOBYWEEK)

    return WCOBYWEEK
    
