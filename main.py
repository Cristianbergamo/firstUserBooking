import os
import pickle
import random

import numpy as np
import pandas as pd
from imblearn import over_sampling
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
from sklearn.multiclass import OneVsOneClassifier

pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

MMS = MinMaxScaler()

NORMALIZER = (MinMaxScaler(), StandardScaler(), RobustScaler())
MODELS = (
    GradientBoostingClassifier(n_estimators=150, max_features='auto'), RandomForestClassifier(),
    SVC(gamma='auto', class_weight='balanced'))
PARAMETERS = ({'min_samples_split': [2, 5, 8],
               'max_depth': [3, 5, 8],
               'learning_rate': [0.08, 0.1, 0.12]},
              {'class_weight': ['balanced'],
               'n_estimators': [20],
               'n_jobs': [2],
               'criterion': ['gini', 'entropy'],
               'max_depth': [None, 1, 3, 7],
               'min_samples_split': [3, 10, 20],
               'min_samples_leaf': [1, 4, 10, 21],
               'max_features': [0.1, 0.3, 0.5, 0.8, 1.],
               'max_leaf_nodes': [None, 3, 5, 10],

               },
              {'C': [1, 0.9, 1.1],
               'kernel': ['rbf', 'poly', 'sigmoid'],
               'degree': [2, 3, 4]
               })
NAMES = ['GRADIENTBOOSTING', 'RANDOMFOREST', 'SVM']

NDF_GRADIENT = GradientBoostingClassifier(n_estimators=100, max_features='auto', verbose=2, max_depth=10)
NDF_FOREST = RandomForestClassifier(criterion='entropy', max_depth=10, max_features='auto', n_estimators=200,
                                    class_weight='balanced')
NDF_SVM = SVC(gamma='auto', class_weight='balanced', C=1.1, degree=2, kernel='rbf')

US_GRADIENT = GradientBoostingClassifier(learning_rate=0.08, max_depth=3, min_samples_split=2, max_features='auto',
                                         n_estimators=150)
US_FOREST = RandomForestClassifier(criterion=None, max_depth=None, max_features=None, n_estimators=None,
                                   class_weight=None)
US_SVM = SVC(gamma=None, class_weight=None, C=None, degree=None, kernel=None)

ABROAD_GRADIENT = GradientBoostingClassifier(learning_rate=0.01, max_depth=1, max_features=0.5,
                                             max_leaf_nodes=3, min_samples_leaf=21, min_samples_split=3,
                                             n_estimators=30)
ABROAD_FOREST = RandomForestClassifier(criterion='gini', max_depth=10, max_features=0.5, max_leaf_nodes=30,
                                       min_samples_leaf=21, min_samples_split=30, n_estimators=100,
                                       class_weight='balanced', verbose=0)
ABROAD_SVM = SVC(class_weight='balanced')

MODELS_PATH = os.path.join(os.getcwd(), 'models')


def elaborateOnlineActivityStatistics(users=None):
    sessions = mapDevices(pd.read_csv(os.path.join(os.getcwd(), 'data', 'sessions.csv')))
    sessions.replace('-unknown-', np.nan, inplace=True)
    sessions.replace('untracked', np.nan, inplace=True)
    sessions.dropna(subset=['secs_elapsed'], inplace=True)

    total_action = sessions.pivot_table(values='secs_elapsed', index='user_id', aggfunc='count').fillna(0)
    nunique_a = sessions.pivot_table(values='action', index='user_id', aggfunc=pd.Series.nunique).fillna(0)
    nunique_a_type = sessions.pivot_table(values='action_type', index='user_id', aggfunc=pd.Series.nunique).fillna(0)
    nunique_a_det = sessions.pivot_table(values='action_detail', index='user_id', aggfunc=pd.Series.nunique).fillna(0)
    nunique_device = sessions.pivot_table(values='device_type', index='user_id', aggfunc=pd.Series.nunique).fillna(0)

    action_frequency = sessions.pivot_table(values='secs_elapsed', index='user_id', columns=['action'],
                                            aggfunc='count').fillna(0)
    action_detail_frequency = sessions.pivot_table(values='secs_elapsed', index='user_id', columns=['action_detail'],
                                                   aggfunc='count').fillna(0)
    action_type_frequency = sessions.pivot_table(values='secs_elapsed', index='user_id', columns=['action_type'],
                                                 aggfunc='count').fillna(0)
    device_type_frequency = sessions.pivot_table(values='secs_elapsed', index='user_id', columns=['device_type'],
                                                 aggfunc='count').fillna(0)

    sessions_total_time = sessions.pivot_table(values='secs_elapsed', index='user_id', aggfunc='sum').fillna(0)
    sessions_median_time = sessions.pivot_table(values='secs_elapsed', index='user_id', aggfunc='median').fillna(0)
    sessions_min_time = sessions.pivot_table(values='secs_elapsed', index='user_id', aggfunc='min').fillna(0)
    sessions_max_time = sessions.pivot_table(values='secs_elapsed', index='user_id', aggfunc='max').fillna(0)

    action_type_total_time = sessions.pivot_table(values='secs_elapsed', columns='action_type', index='user_id',
                                                  aggfunc='sum').fillna(0)
    action_type_median_time = sessions.pivot_table(values='secs_elapsed', columns='action_type', index='user_id',
                                                   aggfunc='median').fillna(0)
    action_type_min_time = sessions.pivot_table(values='secs_elapsed', columns='action_type', index='user_id',
                                                aggfunc='min').fillna(0)
    action_type_max_time = sessions.pivot_table(values='secs_elapsed', columns='action_type', index='user_id',
                                                aggfunc='max').fillna(0)

    device_type_total_time = sessions.pivot_table(values='secs_elapsed', columns='device_type', index='user_id',
                                                  aggfunc='sum').fillna(0)
    device_type_median_time = sessions.pivot_table(values='secs_elapsed', columns='device_type', index='user_id',
                                                   aggfunc='median').fillna(0)
    device_type_min_time = sessions.pivot_table(values='secs_elapsed', columns='device_type', index='user_id',
                                                aggfunc='min').fillna(0)
    device_type_max_time = sessions.pivot_table(values='secs_elapsed', columns='device_type', index='user_id',
                                                aggfunc='max').fillna(0)

    users = users.merge(total_action, how='left', left_on='id', right_index=True)
    users = users.merge(nunique_a, how='left', left_on='id', right_index=True)
    users = users.merge(nunique_a_type, how='left', left_on='id', right_index=True)
    users = users.merge(nunique_a_det, how='left', left_on='id', right_index=True)
    users = users.merge(nunique_device, how='left', left_on='id', right_index=True)
    users = users.merge(action_frequency, how='left', left_on='id', right_index=True)
    users = users.merge(action_detail_frequency, how='left', left_on='id', right_index=True)
    users = users.merge(action_type_frequency, how='left', left_on='id', right_index=True)
    users = users.merge(device_type_frequency, how='left', left_on='id', right_index=True)
    users = users.merge(sessions_total_time, how='left', left_on='id', right_index=True)
    users = users.merge(sessions_median_time, how='left', left_on='id', right_index=True)
    users = users.merge(sessions_min_time, how='left', left_on='id', right_index=True)
    users = users.merge(sessions_max_time, how='left', left_on='id', right_index=True)
    users = users.merge(action_type_total_time, how='left', left_on='id', right_index=True)
    users = users.merge(action_type_median_time, how='left', left_on='id', right_index=True)
    users = users.merge(action_type_min_time, how='left', left_on='id', right_index=True)
    users = users.merge(action_type_max_time, how='left', left_on='id', right_index=True)
    users = users.merge(device_type_total_time, how='left', left_on='id', right_index=True)
    users = users.merge(device_type_median_time, how='left', left_on='id', right_index=True)
    users = users.merge(device_type_min_time, how='left', left_on='id', right_index=True)
    users = users.merge(device_type_max_time, how='left', left_on='id', right_index=True)

    return users


def categoricalMapping(users):
    first_browser_dict = {'SeaMonkey': 'other',
                          'Mozilla': 'other',
                          'RockMelt': 'other',
                          'IceDragon': 'other',
                          'Opera Mini': 'other',
                          'Googlebot': 'other',
                          'Outlook 2007': 'other',
                          'TenFourFox': 'other',
                          'Avant Browser': 'other',
                          'TheWorld Browser': 'other',
                          'CoolNovo': 'other',
                          'Iron': 'other',
                          'Pale Moon': 'other',
                          'IceWeasel': 'other',
                          'Yandex.Browser': 'other',
                          'SiteKiosk': 'other',
                          'BlackBerry Browser': 'other',
                          'Apple Mail': 'other',
                          'Maxthon': 'other',
                          'Sogou Explorer': 'other',
                          'Mobile Firefox': 'other',
                          'IE Mobile': 'other',
                          'Chromium': 'other',
                          'AOL Explorer': 'other',
                          'Silk': 'other',
                          'Opera': 'other',
                          'Android Browser': 'Android Browser',
                          'Chrome Mobile': 'Chrome Mobile',
                          'IE': 'IE',
                          'Mobile Safari': 'Mobile Safari',
                          'Firefox': 'Firefox',
                          'nan': 'nan',
                          'Safari': 'Safari',
                          'Chrome': 'Chrome'}
    affiliate_provider_dict = {'daum': 'minority',
                               'craigslist': 'minority',
                               'meetup': 'minority',
                               'baidu': 'minority',
                               'yandex': 'minority',
                               'naver': 'minority',
                               'gsp': 'minority',
                               'vast': 'minority',
                               'facebook-open-graph': 'facebook-open-graph',
                               'email-marketing': 'email-marketing',
                               'yahoo': 'yahoo',
                               'padmapper': 'padmapper',
                               'facebook': 'facebook',
                               'bing': 'bing',
                               'other': 'other',
                               'google': 'google',
                               'direct': 'direct', }

    df = users.copy()
    df.loc[:, 'first_browser'] = df.loc[:, 'first_browser'].map(first_browser_dict)
    df.loc[:, 'affiliate_povider'] = df.loc[:, 'affiliate_provider'].map(affiliate_provider_dict)

    return df


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


def categoricalTransformation(users, ohe=None):
    categorical = ['gender', 'language', 'affiliate_channel', 'affiliate_provider',
                   'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
    users_mapped = categoricalMapping(users)
    if ohe is None:
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        dummy_users = pd.DataFrame(ohe.fit_transform(users_mapped[categorical].fillna('nan')), index=users.index)
        with open(os.path.join(MODELS_PATH, 'ohe.bin'), 'wb') as ohe_file:
            pickle.dump(ohe, ohe_file)
            ohe_file.close()
    else:
        dummy_users = pd.DataFrame(ohe.transform(users_mapped[categorical].fillna('nan')), index=users.index)

    return ohe, dummy_users


def numericalToPca(users, pca=None, ss=None):
    if ss is None:
        ss = StandardScaler()
        users_standardized = np.array(users)
        with open(os.path.join(MODELS_PATH, 'ss_numerical_encoding.bin'), 'wb') as f:
            pickle.dump(ss, f)
            f.close()
    else:
        users_standardized = np.array(users)

    if pca is None:
        pca = PCA(360)
        numerical_to_pca = pd.DataFrame(np.array(users_standardized), index=users.index)
        with open(os.path.join(MODELS_PATH, 'pca_numerical_encoding.bin'), 'wb') as f:
            pickle.dump(pca, f)
            f.close()
    else:
        numerical_to_pca = pd.DataFrame(np.array(users_standardized), index=users.index)

    return ss, pca, numerical_to_pca


def fitAndSaveTrees(X_train, y_train, model, model_name, resample=False):
    if resample:
        resampler = over_sampling.BorderlineSMOTE(kind='borderline-2')
        X_train_r, y_train_r = resampler.fit_resample(X_train, y_train)

        model.fit(X_train_r, y_train_r)
    else:
        model.fit(X_train, y_train)

    with open(os.path.join(os.getcwd(), 'models', model_name + r'.bin'), 'wb') as model_file:
        pickle.dump(model, model_file)
        model_file.close()

    return model


def fitAndSaveSVM(X_train, y_train, model, model_name, resample=False):
    mms = MinMaxScaler()
    X_train_scaled = mms.fit_transform(X_train)

    if resample:

        resampler = SMOTE()
        X_train_r, y_train_r = resampler.fit_resample(X_train_scaled, y_train)

        model.fit(X_train_r, y_train_r)

    else:
        model.fit(X_train_scaled, y_train)

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


def multiModelPrediction(users):
    countries = {0: 'NDF', 1: 'US', 2: 'AU', 3: 'CA', 4: 'DE', 5: 'ES', 6: 'FR', 7: 'GB', 8: 'IT', 9: 'NL', 10: 'PT',
                 11: 'other'}

    with open(os.path.join(MODELS_PATH, 'NDF_OVR.bin'), 'rb') as f:
        ndf = pickle.load(f)
        f.close()
    with open(os.path.join(MODELS_PATH, 'US_OVR.bin'), 'rb') as f:
        us = pickle.load(f)
        f.close()
    with open(os.path.join(MODELS_PATH, 'AU_OVR.bin'), 'rb') as f:
        au = pickle.load(f)
        f.close()
    with open(os.path.join(MODELS_PATH, 'CA_OVR.bin'), 'rb') as f:
        ca = pickle.load(f)
        f.close()
    with open(os.path.join(MODELS_PATH, 'DE_OVR.bin'), 'rb') as f:
        de = pickle.load(f)
        f.close()
    with open(os.path.join(MODELS_PATH, 'ES_OVR.bin'), 'rb') as f:
        es = pickle.load(f)
        f.close()
    with open(os.path.join(MODELS_PATH, 'FR_OVR.bin'), 'rb') as f:
        fr = pickle.load(f)
        f.close()
    with open(os.path.join(MODELS_PATH, 'GB_OVR.bin'), 'rb') as f:
        gb = pickle.load(f)
        f.close()
    with open(os.path.join(MODELS_PATH, 'IT_OVR.bin'), 'rb') as f:
        it = pickle.load(f)
        f.close()
    with open(os.path.join(MODELS_PATH, 'NL_OVR.bin'), 'rb') as f:
        nl = pickle.load(f)
        f.close()
    with open(os.path.join(MODELS_PATH, 'PT_OVR.bin'), 'rb') as f:
        pt = pickle.load(f)
        f.close()
    with open(os.path.join(MODELS_PATH, 'other_OVR.bin'), 'rb') as f:
        other = pickle.load(f)
        f.close()

    predictions_prob = np.column_stack((ndf.predict_proba(users)[:, 1],
                                        us.predict_proba(users)[:, 1],
                                        au.predict_proba(users)[:, 1],
                                        ca.predict_proba(users)[:, 1],
                                        de.predict_proba(users)[:, 1],
                                        es.predict_proba(users)[:, 1],
                                        fr.predict_proba(users)[:, 1],
                                        gb.predict_proba(users)[:, 1],
                                        it.predict_proba(users)[:, 1],
                                        nl.predict_proba(users)[:, 1],
                                        pt.predict_proba(users)[:, 1],
                                        other.predict_proba(users)[:, 1]))

    predictions = pd.Series(np.argmax(predictions_prob, axis=1)).map(countries)
    ndf_prediction = predictions

    return ndf_prediction


def validate(X_train=None, y_train=None, X_test=None, y_test=None, target=None, normalizer=None, model=None,
             parameters=None, resampler=None, model_name=None):
    if normalizer is not None:
        X_train = normalizer.fit_transform(X_train)
        X_test = normalizer.transform(X_test)

    if resampler is not None:
        X_train_r, y_train_r = resampler.fit_resample(X_train, y_train)
        shuffled_index = random.sample(range(len(X_train_r)), len(X_train_r))
        X_train = X_train_r[shuffled_index]
        y_train = y_train_r[shuffled_index]

    gs = GridSearchCV(model, parameters, cv=3, verbose=2, scoring='balanced_accuracy', iid=False, refit=True, n_jobs=4)
    gs.fit(X_train, y_train)
    with open(os.path.join(os.getcwd(), 'validationLogs',
                           '%s_%s_cross_validation_log.txt' % (model_name, target)), 'a') as logging_file:
        logging_file.write(model_name + ' - ' + 'TARGET: %s \n' % target)
        logging_file.write(str(gs.best_params_) + '\n')
        logging_file.write('Best score - ' + str(gs.best_score_) + '\n')
        logging_file.write('Training Confusion Matrix\n%s\n' % confusion_matrix(y_train, gs.predict(X_train)))
        logging_file.write('Test Confusion Matrix\n%s\n' % confusion_matrix(y_test, gs.predict(X_test)))
        logging_file.write('Test - score\n%s \n\n' % balanced_accuracy_score(y_test, gs.predict(X_test), adjusted=True))
        logging_file.close()
    return


if __name__ == '__main__':
    # users = pd.read_csv(os.path.join(os.getcwd(), 'data', 'train_users.csv'))

    ''' AGE FIXING '''
    # users.replace('-unknown-', np.nan, inplace=True)
    # age = np.array(users.age)
    # np.place(age, age >= 110, np.nan)
    # np.place(age, age <= 14, np.nan)
    # np.place(age, age >= 80, 80)
    # users.loc[:, 'age'] = age
    # users.age.fillna(users.age.median(), inplace=True)

    ''' DATETIME ENGINEERING '''
    # users.loc[:, 'timestamp_first_active'] = pd.to_datetime(users['timestamp_first_active'].astype(str).str[:8])
    # users.loc[:, 'date_account_created'] = pd.to_datetime(users['date_account_created'])
    # users.insert(1, 'time_gap_after_creation', (users.timestamp_first_active - users.date_account_created).dt.days)
    # users.insert(1, 'week_created', users.loc[:, 'date_account_created'].dt.week)
    # users.insert(1, 'week_first_active', users.loc[:, 'timestamp_first_active'].dt.week)
    # users.drop(['date_account_created', 'date_first_booking'], axis=1, inplace=True)

    ''' SESSIONS STATISTICS '''
    # users_session = elaborateOnlineActivityStatistics(users)

    ''' CREATE TRAIN/TEST DATASETS  '''
    # Y = np.array(users_session['country_destination'])
    # train_df, test_df, _, _ = train_test_split(users_session, Y, test_size=0.10, random_state=42)
    # ohe, train_categorical_cols_encoded = categoricalTransformation(train_df, ohe=None)
    # train_df = pd.concat(
    #     (train_df[['id', 'country_destination', 'timestamp_first_active', 'time_gap_after_creation','week_first_active','week_created','age','signup_flow']],
    #      train_categorical_cols_encoded, train_df.iloc[:, 17:].fillna(0)), axis=1)
    # _, test_categorical_cols_encoded = categoricalTransformation(test_df, ohe=ohe)
    # test_df = pd.concat(
    #     (test_df[['id', 'country_destination', 'timestamp_first_active', 'time_gap_after_creation', 'week_first_active','week_created','age','signup_flow']],
    #      test_categorical_cols_encoded, test_df.iloc[:, 17:].fillna(0)),
    #     axis=1)

    '''SAVE TRAIN/TEST DF'''
    # with open('train', 'wb') as f:
    #     pickle.dump(train_df, f)
    #     f.close()
    # with open('test', 'wb') as f:
    #     pickle.dump(test_df, f)
    #     f.close()

    '''LOAD TRAIN/TEST DF'''
    with open('train', 'rb') as f:
        train_df = pickle.load(f)
        y_train = np.array(train_df['country_destination'])
        X_train = train_df.iloc[:, 2:]
        f.close()
    with open('test', 'rb') as f:
        test_df = pickle.load(f)
        y_test = np.array(test_df['country_destination'])
        X_test = test_df.iloc[:, 2:]
        f.close()
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    ''' SPLIT Xs INTO SESSION/NO SESSIONS '''
    vt1 = VarianceThreshold()
    vt2 = VarianceThreshold()
    train_mask2014 = np.array(X_train.timestamp_first_active > pd.datetime(2013, 12, 31))
    test_mask2014 = np.array(X_test.timestamp_first_active > pd.datetime(2013, 12, 31))
    X_train_sessions = vt1.fit_transform(X_train[train_mask2014].iloc[:, [i for i in range(4, 683)]])
    X_test_sessions = vt1.transform(X_test.iloc[:, [i for i in range(4, 683)]])
    X_train_no_sessions = vt2.fit_transform(X_train.iloc[:, [i for i in range(1, 91)]])
    X_test_no_sessions = vt2.transform(X_test.iloc[:, [i for i in range(1, 91)]])

    ''' CROSS VALIDATION 2014 ONLY'''
    # estimator = XGBClassifier()
    # param_grid = {'n_estimators': [100],
    #               'objective': ['multi:softprob'],
    #               'learning_rate': [0.05, 0.1, 0.15],
    #               'min_child_weigth': [1, 2, 4, 8],
    #               'max_depth': [3, 6, 13, 21],
    #               'gamma': [0, 0.1, 0.2, 0.4, 0.8],
    #               'max_delta_step': [0, 1, 2, 4, 8],
    #               'subsample': [0.8, 0.9, 1],
    #               'colsample_bytree': [0.5, 0.8, 1],
    #               'reg_lambda': [1, 2, 4],
    #               'scale_pos_weight': [0.1, 0.2, 0.4, 0.8, 1]
    #               }

    # param_grid = {'n_estimators': [100],
    #               'objective': ['multi:softprob'],
    #               'learning_rate': [0.05, 0.15],
    #               'min_child_weigth': [1, 8],
    #               'max_depth': [3, 21],
    #               'gamma': [0.1, 0.8],
    #               'max_delta_step': [0, 8],
    #               'subsample': [0.8, 1],
    #               'colsample_bytree': [0.5, 1],
    #               'reg_lambda': [1, 4],
    #               'scale_pos_weight': [0.1, 1]
    #               }
    #
    # validate(model=estimator, X_train=X_train_sessions, y_train=y_train[train_mask2014], target='2014',
    #          model_name='XGBOOST', X_test=X_test_sessions[test_mask2014], y_test=y_test[test_mask2014],
    #          parameters=param_grid)


    ''' 2Ways SOLUTION'''
    # session_model = XGBClassifier(n_estimators=100, max_depth=10,objective='multi:softprob',n_jobs=-1)
    # session_model.fit(X_train_sessions, y_train[train_mask2014])
    # session_proba = session_model.predict_proba(X_test_sessions[test_mask2014])

    # no_session_model = XGBClassifier(n_estimators=5, max_depth=3)
    # no_session_model.fit(X_train_no_sessions, y_train)
    # no_session_proba = no_session_model.predict_proba(X_test_no_sessions)

    # session_weight = len(X_train_sessions) / float(len(X_train))
    # no_session_weight = 1. - session_weight

    # prediciton_proba_sessions = (session_proba * session_weight) + (no_session_proba[test_mask2014] * no_session_weight)
    # prediciton_proba_no_sessions = no_session_proba[~test_mask2014]
    # prediction_proba = np.concatenate((prediciton_proba_no_sessions, prediciton_proba_sessions), axis=0)
    # prediction = np.argmax(prediction_proba, axis=1)

    # y_test_reindexed = np.concatenate((y_test[~test_mask2014], y_test[test_mask2014]))
    # print(confusion_matrix(y_test_reindexed, prediction))
    # print(balanced_accuracy_score(y_test_reindexed, prediction, adjusted=True))
    # print(accuracy_score(y_test_reindexed, prediction))

    ''' analysis of single models - part of 2ways analysis'''
    # predicted_session = session_model.predict(X_test_sessions[test_mask2014])
    # print(confusion_matrix(y_test[test_mask2014], predicted_session))
    # print(balanced_accuracy_score(y_test[test_mask2014], predicted_session, adjusted=True))
    # print(accuracy_score(y_test[test_mask2014], predicted_session))

    # predicted_no_session = no_session_model.predict(X_test_no_sessions)
    # print(confusion_matrix(y_test, predicted_no_session))
    # print(balanced_accuracy_score(y_test, predicted_no_session, adjusted=True))
    # print(accuracy_score(y_test, predicted_no_session))

    '''cross validation single country for OVR'''
    # estimator = XGBClassifier()
    # y_train_country = (le.inverse_transform(y_train) == 'DE').astype(int)
    # y_test_country = (le.inverse_transform(y_test) == 'DE').astype(int)
    # param_grid = {'n_estimators': [100],
    #               'objective': ['binary:logistic'],
    #               'learning_rate': [0.05, 0.15],
    #               'min_child_weigth': [1, 8],
    #               'max_depth': [3, 21],
    #               'gamma': [0.1, 0.8],
    #               'max_delta_step': [0, 8],
    #               'subsample': [0.8, 1],
    #               'colsample_bytree': [0.5, 1],
    #               'reg_lambda': [1, 4],
    #               'scale_pos_weight': [0.1, 1]
    #               }
    # validate(X_train_no_sessions, y_train_country, X_test_no_sessions, y_test_country, 'DE', None, estimator, param_grid,
    #          None, 'OVR_XGB_NOSESSION')

    '''cross validation single country for OVO'''
    estimator = XGBClassifier()
    y_train_countries = le.inverse_transform(y_train)
    y_test_countries = le.inverse_transform(y_test)
    de_es_mask_train = np.isin(y_train_countries, ['DE', 'NDF'])
    de_es_mask_test = np.isin(y_test_countries, ['DE', 'NDF'])
    y_train_bool = (y_train_countries[de_es_mask_train] == 'DE').astype(int)
    y_test_bool = (y_test_countries[de_es_mask_test] == 'DE').astype(int)
    param_grid = {'n_estimators': [100],
                  'objective': ['binary:logistic'],
                  'learning_rate': [0.05, 0.15],
                  'min_child_weigth': [1, 8],
                  'max_depth': [3, 21],
                  'gamma': [0.1, 0.8],
                  'max_delta_step': [0, 8],
                  'subsample': [0.8, 1],
                  'colsample_bytree': [0.5, 1],
                  'reg_lambda': [1, 4],
                  'scale_pos_weight': [0.1, 1]
                  }
    validate(X_train_no_sessions[de_es_mask_train], y_train_bool,
             X_test_no_sessions[de_es_mask_test], y_test_bool, 'DE', None,
             estimator, param_grid, None, 'OVO_XGB_NOSESSIONS')

    ''' OVO SESSION FIT '''
    # estimator = XGBClassifier(n_estimators=100,objective='binary:logistic',learning_rate=0.15,min_child_weight=1,
    #                           max_depth=21,gamma=0.8,max_delta_step=0,subsample=1,colsample_bytree=0.4,reg_lambda=4,
    #                           scale_pos_weight=1.1,verbosity = 2)
    # model = OneVsOneClassifier(estimator = estimator,n_jobs=4)
    # model.fit(X_train_sessions,y_train[train_mask2014])
    # with open('ovo_sessions.bin','wb') as f:
    #     pickle.dump(model,f)
    #     f.close()
    # prediction = model.predict(X_test_sessions[test_mask2014])
    # print(confusion_matrix(y_test[test_mask2014],prediction))
    # print(balanced_accuracy_score(y_test[test_mask2014],prediction))

# TODO: provare cross validation per no session
# TODO: valutare risultati cross validation ovr session e se in caso affinare la param grids
# TODO: ripetere il  procedimento fatto per OVR anche per OVO
