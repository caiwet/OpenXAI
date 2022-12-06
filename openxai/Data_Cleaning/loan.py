import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from data_cleaning_utils import convert_categorical_cols


label_col = 'loan_repaid'
loan_feature_types = ['d', 'd', 'c', 'c', 'd', 'd', 'c' ]

def main():
    accepted = pd.read_csv('../data_raw/accepted_2007_to_2018Q4.csv')
    accepted = accepted[(accepted['loan_status'] == 'Fully Paid') | \
                        (accepted['loan_status'] == 'Charged Off')]
    accepted['loan_repaid'] = accepted['loan_status'].map({'Fully Paid':1,'Charged Off':0})
    accepted['avg_fico'] = (accepted['fico_range_low'] + accepted['fico_range_high'])/2
    accepted['credit_score'] = pd.cut(accepted['avg_fico'], [0,580,669,739,799,10000], 
                                  labels=['Poor', 'Fair', 'Good', 'Very Good', 'Exceptional'])
    
    cols = ['loan_amnt', 'annual_inc', 'credit_score', 'home_ownership', 
        'int_rate', 'installment', 'term']
    data_df = accepted[cols]
    y_col = accepted['loan_repaid']
    
    cols = np.array(data_df.columns)
    cat_cols = cols[np.argwhere(np.array(loan_feature_types) == 'c').flatten()]

    one_hot, feature_metadata = convert_categorical_cols(data = data_df, 
                                                         feature_types = loan_feature_types,
                                                         original_columns = cols, 
                                                         columns = cat_cols)

    # append the label column to the end of the dataframe
    one_hot[label_col] = y_col
    X_train, X_test = train_test_split(one_hot, stratify=one_hot['loan_repaid'], 
                                       test_size=0.20, random_state = 0)
    X_train.to_csv('../data/Lending_Club/loan_train.csv', index=False)
    X_test.to_csv('../data/Lending_Club/loan_test.csv', index=False)
    
    
if __name__ == "__main__":

    main()