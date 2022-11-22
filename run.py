import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis', 'model'. 
    
    `main` runs the targets in order of data=>analysis=>model.
    '''

    if 'test' in targets:
        all_activity_data = ['data/patient_1/patient_1_activity.csv', 'data/patient_2/patient_2_activity.csv',
                    'data/patient_3/patient_3_activity.csv']
        test_data = ['data/test/testdata/test_data.csv']
        li = []
        for filename in test_data:
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)

        data = pd.concat(li, axis=0, ignore_index=True)
        data = data\
        [['summary_date',
        'score_stay_active',
        'score_move_every_hour',
        'score_meet_daily_targets',
        'score_training_frequency',
        'score_training_volume',
        'score_recovery_time']]
        def get_weekday_end(weekday):
            if weekday in [5,6,7]:
                return 'weekend'
            else:
                return 'weekday'
    
        data['weekday'] = pd.to_datetime(data['summary_date']).apply(lambda x: get_weekday_end(x.weekday()))
        data['weekday'] = LabelEncoder().fit_transform(data['weekday'])

        pipeline = Pipeline([
            ('normalizer', StandardScaler()), #Step1 - normalize data
            ('clf', LogisticRegression()) #step2 - classifier
        ])
        X_train, y_train = data.iloc[:,1:-1].values, data['weekday']
        
    return


if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)