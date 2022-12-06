import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    res1 = []
    with open('../data_raw/ICPSR_1978/DS0001/08987-0001-Data.txt') as f:
        for line in f:
            row = [line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8], line[9], 
                  line[10:12], line[12:14], line[14:16], line[16:19], line[19:22], line[22:24], line[24], line[25:27], line[27]]
            row = [int(i) for i in row]
            res1.append(row)

    df1978 = pd.DataFrame(res1, columns=['white', 'alchy', 'junky', 'super', 'married', 'felon', 'workrel', 'propty', 
                                  'person', 'male', 'priors', 'school', 'rule', 'age', 'tservd', 'follow', 'recid', 
                                  'time', 'file'])
    X_train, X_test = train_test_split(df1978, stratify=df1978['recid'], 
                                           test_size=0.20, random_state = 0)
    X_train.to_csv('../data/RCDV/rcdv1978_train.csv', index=False)
    X_test.to_csv('../data/RCDV/rcdv1978_test.csv', index=False)


    res2 = []
    with open('../data_raw/ICPSR_1980/DS0002/08987-0002-Data.txt') as f:
        for line in f:
            row = [line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8], line[9], 
                  line[10:12], line[12:14], line[14:16], line[16:19], line[19:22], line[22:24], line[24], line[25:27], line[27]]
            row = [int(i) for i in row]
            res2.append(row)


    df1980 = pd.DataFrame(res2, columns=['white', 'alchy', 'junky', 'super', 'married', 'felon', 'workrel', 'propty', 
                                  'person', 'male', 'priors', 'school', 'rule', 'age', 'tservd', 'follow', 'recid', 
                                  'time', 'file'])
    X_train, X_test = train_test_split(df1980, stratify=df1980['recid'], 
                                           test_size=0.20, random_state = 0)
    X_train.to_csv('../data/RCDV/rcdv1980_train.csv', index=False)
    X_test.to_csv('../data/RCDV/rcdv1980_test.csv', index=False)
    
if __name__ == "__main__":

    main()