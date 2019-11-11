import os
import pickle
import random

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.svm import SVC

pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

NORMALIZER = (MinMaxScaler(), StandardScaler(), RobustScaler())
MODELS = (GradientBoostingClassifier(), RandomForestClassifier(class_weight='balanced'),
          SVC(gamma='auto', class_weight='balanced'))
PARAMETERS = ({'learning_rate': [0.1, 0.05, 0.15],
               'n_estimators': [50, 100, 200],
               'max_depth': [5, 10, 15],
               'max_features': [None, 'auto']
               },
              {
                  'n_estimators': [50, 100, 200],
                  'criterion': ['gini', 'entropy'],
                  'max_depth': [None, 10, 20],
                  'max_features': ['auto', None],
              },
              {'C': [1, 0.9, 1.1],
               'kernel': ['rbf', 'poly', 'sigmoid'],
               'degree': [2, 3, 4]
               })
NAMES = ['GRADIENTBOOSTING', 'RANDOMFOREST', 'SVM']

NDF_GRADIENT = GradientBoostingClassifier(learning_rate=0.1, max_depth=5, max_features=None, n_estimators=200)
NDF_FOREST = RandomForestClassifier(criterion='entropy', max_depth=None, max_features='auto', n_estimators=200,
                                    class_weight='balanced')
NDF_SVM = SVC(gamma='auto', class_weight='balanced', C=1.1, degree=2, kernel='rbf')


def validate(X_train=None, y_train=None, X_test=None, y_test=None, target_country=None, normalizer=None, model=None,
             parameters=None, resampler=None,
             model_name=None):
    if normalizer is not None:
        X_train = normalizer.fit_transform(X_train)
        X_test = normalizer.transform(X_test)

    if resampler is not None:
        X_train_r, y_train_r = resampler.fit_resample(X_train, y_train)
        shuffled_index = random.sample(range(len(X_train_r)), len(X_train_r))
        X_train = X_train_r[shuffled_index]
        y_train = y_train_r[shuffled_index]

    gs = GridSearchCV(model, parameters, cv=5, verbose=2, scoring='f1', iid=False, refit=True)
    gs.fit(X_train, y_train)
    with open(os.path.join(os.getcwd(), 'validationLogs',
                           '%s_%s_cross_validation_log.txt' % (model_name, target_country)), 'a') as logging_file:
        logging_file.write(model_name + ' - ' + 'TARGET COUNTRY: %s \n' % target_country)
        logging_file.write(str(gs.best_params_) + '\n')
        logging_file.write('Best F1 - ' + str(gs.best_score_) + '\n')
        logging_file.write('Training Confusion Matrix\n%s\n' % confusion_matrix(y_train, gs.predict(X_train)))
        logging_file.write('Test Confusion Matrix\n%s\n' % confusion_matrix(y_test, gs.predict(X_test)))
        logging_file.write('Test - F1 score\n%s \n\n' % f1_score(y_test, gs.predict(X_test)))
        logging_file.close()
    return


def elaborateOnlineActivityStatistics(users=None):
    sessions = pd.read_csv(os.path.join(os.getcwd(), 'data', 'sessions.csv'))
    sessions_users = mapDevices(sessions[sessions['user_id'].isin(users['id'])])
    del sessions
    sessions_users['min_elapsed'] = sessions_users['secs_elapsed'] / 60
    sessions_users.drop(['secs_elapsed'], axis=1, inplace=True)

    pivot_sum_action_time = sessions_users.pivot_table(values='min_elapsed', index='user_id', columns='action_type',
                                                       aggfunc='sum').fillna(0)
    pivot_sum_action_time.columns = [x + '_total_time' for x in pivot_sum_action_time.columns]
    pivot_sum_action_count = sessions_users.pivot_table(values='min_elapsed', index='user_id', columns='action_type',
                                                        aggfunc='count').fillna(0)
    pivot_sum_action_count.columns = [x + '_total_events' for x in pivot_sum_action_count.columns]
    pivot_action_heterogeneity = sessions_users.pivot_table(values='action_type', index='user_id',
                                                            aggfunc=lambda x: len(np.unique(x.astype(str))))
    pivot_action_heterogeneity.columns = ['action_heterogeneity']

    pivot_sum_device_time = sessions_users.pivot_table(values='min_elapsed', index='user_id', columns='device_type',
                                                       aggfunc='sum').fillna(0)
    pivot_sum_device_time.columns = [x + '_total_time' for x in pivot_sum_device_time.columns]
    pivot_sum_device_count = sessions_users.pivot_table(values='min_elapsed', index='user_id', columns='device_type',
                                                        aggfunc='count').fillna(0)
    pivot_sum_device_count.columns = [x + '_total_events' for x in pivot_sum_device_count.columns]
    pivot_device_heterogeneity = sessions_users.pivot_table(values='device_type', index='user_id',
                                                            aggfunc=lambda x: len(np.unique(x.astype(str))))
    pivot_device_heterogeneity.columns = ['device_heterogeneity']

    users = users.merge(pivot_sum_action_time, how='inner', left_on='id', right_index=True)
    users = users.merge(pivot_sum_action_count, how='inner', left_on='id', right_index=True)
    users = users.merge(pivot_action_heterogeneity, how='inner', left_on='id', right_index=True)
    users = users.merge(pivot_sum_device_time, how='inner', left_on='id', right_index=True)
    users = users.merge(pivot_sum_device_count, how='inner', left_on='id', right_index=True)
    users = users.merge(pivot_device_heterogeneity, how='inner', left_on='id', right_index=True)

    return users


def mapDevices(sessions):
    device_mapping = {'Windows Desktop': 'Desktop',
                      '-unknown-': 'other',
                      'Mac Desktop': 'Desktop',
                      'Android Phone': 'Phone',
                      'iPhone': 'Phone',
                      'iPad Tablet': 'Tablet',
                      'Android App Unknown Phone/Tablet': 'unknown',
                      'Linux Desktop': 'Desktop',
                      'Tablet': 'Tablet',
                      'Chromebook': 'Desktop',
                      'Blackberry': 'Phone',
                      'iPodtouch': 'Phone',
                      'Windows Phone': 'Phone',
                      'Opera Phone': 'Phone'
                      }
    sessions.loc[:, ['device_type']] = sessions['device_type'].map(device_mapping)

    return sessions


def categoricalToPca(users):
    categorical = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider',
                   'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
    dummy_users = pd.get_dummies(users[categorical].fillna('nan'))
    pca = PCA(3)
    categoricalToPca = pd.DataFrame(pca.fit_transform(dummy_users))

    return categoricalToPca


def fitAndSaveTrees(X_train, y_train, model, model_name):
    resampler = SMOTE()
    X_train_r, y_train_r = resampler.fit_resample(X_train, y_train)

    model.fit(X_train_r, y_train_r)
    with open(os.path.join(os.getcwd(), 'models', model_name + r'.bin'), 'wb') as model_file:
        pickle.dump(model, model_file)
        model_file.close()

    return model


def fitAndSaveSVM(X_train, y_train, model, model_name):
    mms = MinMaxScaler()
    X_train_scaled = mms.fit_transform(X_train)

    resampler = SMOTE()
    X_train_r, y_train_r = resampler.fit_resample(X_train_scaled, y_train)

    model.fit(X_train_r, y_train_r)
    with open(os.path.join(os.getcwd(), 'models', model_name + r'.bin'), 'wb') as model_file:
        pickle.dump(model, model_file)
        model_file.close()

    with open(os.path.join(os.getcwd(), 'models', model_name + r'_mms.bin'), 'wb') as model_file:
        pickle.dump(mms, model_file)
        model_file.close()

    return model


def loadFitModel(model_name):
    with open(os.path.join(os.getcwd(), 'models', model_name + r'.bin'), 'rb') as model_file:
        model = pickle.load(model_file)
        model_file.close()

    return model


if __name__ == '__main__':
    # train_users = pd.read_csv(os.path.join(os.getcwd(), 'data', 'train_users.csv'))
    # train_users.drop('age', axis=1, inplace=True)
    # categorical_cols_encoded = categoricalToPca(train_users)
    # users = pd.concat((train_users[['id', 'country_destination']], categorical_cols_encoded), axis=1)
    #
    # full_df = elaborateOnlineActivityStatistics(users)
    # with open('full_df','wb') as f:
    #     pickle.dump(full_df,f)
    #     f.close()

    with open('full_df', 'rb') as f:
        full_df = pickle.load(f)
        f.close()

    resampler = SMOTE()
    mms = MinMaxScaler()

    Y = np.array(full_df['country_destination'])
    X = full_df.iloc[:, 2:]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    y_train_ndf = (y_train == 'NDF').astype(int)
    y_test_ndf = (y_test == 'NDF').astype(int)

    # '''VALIDATION NDF'''
    # model = 2  # 0:GradientBoosting; 1:RandomForest; 2:SVM
    # validate(X_train=X_train, y_train=y_train_ndf, X_test=X_test, y_test=y_test_ndf, target_country='NDF',
    #          normalizer=mms, model=MODELS[model], parameters=PARAMETERS[model],
    #          resampler=None, model_name=NAMES[model])

    # '''FIT AND SAVE MODELS WITH VALIDATED PARAMETERS FOR NDF'''
    # ndf_gradient = fitAndSaveTrees(X_train,y_train_ndf,NDF_GRADIENT,'ndfgradient')
    # ndf_forest = fitAndSaveTrees(X_train,y_train_ndf,NDF_FOREST,'ndfforest')
    # ndf_svm = fitAndSaveSVM(X_train, y_train_ndf, NDF_SVM, 'ndfsvm')

    '''VALIDATION US'''
    X_train_booked = X_train[y_train != 'NDF']
    y_train_booked = y_train[y_train != 'NDF']
    X_test_booked = X_test[y_test != 'NDF']
    y_test_booked = y_test[y_test != 'NDF']
    y_train_us = (y_train_booked == 'US').astype(int)
    y_test_us = (y_test_booked == 'US').astype(int)
    model = 0
    validate(X_train=X_train_booked, y_train=y_train_us, X_test=X_test_booked, y_test=y_test_us, target_country='US',
             normalizer=None, model=MODELS[model], parameters=PARAMETERS[model],
             resampler=None, model_name=NAMES[model])