import os
import pickle
import random

import numpy as np
import pandas as pd
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
from xgboost.sklearn import XGBClassifier

from ndcgScorer import ndcg_score

pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

MODELS_PATH = os.path.join(os.getcwd(), 'models')


def elaborateOnlineActivityStatistics(users=None):
    sessions = mapDevices(pd.read_csv(os.path.join(os.getcwd(), 'data', 'sessions.csv')))
    sessions.insert(0, 'sessions', np.ones(len(sessions)))
    sessions.replace('-unknown-', np.nan, inplace=True)
    sessions.replace('untracked', np.nan, inplace=True)
    sessions.dropna(subset=['secs_elapsed'], inplace=True)

    '''Statistiche per utente'''
    total_action = sessions.pivot_table(values='secs_elapsed', index='user_id', aggfunc='count')
    nunique_a = sessions.pivot_table(values='action', index='user_id', aggfunc=pd.Series.nunique)
    nunique_a_type = sessions.pivot_table(values='action_type', index='user_id', aggfunc=pd.Series.nunique)
    nunique_a_det = sessions.pivot_table(values='action_detail', index='user_id', aggfunc=pd.Series.nunique)
    nunique_device = sessions.pivot_table(values='device_type', index='user_id', aggfunc=pd.Series.nunique)
    total_time = sessions.pivot_table(values='secs_elapsed', index='user_id', aggfunc='sum')

    '''Statistiche per singolo actionType_device'''
    sessions['action_type_on_device'] = sessions['action_type'].astype(str) + '_' + sessions['device_type'].astype(str)
    total_actions_ad = sessions.pivot_table(values='secs_elapsed', index='user_id', columns=['action_type_on_device'],
                                            aggfunc='count')
    relative_taad = total_actions_ad.div(total_action['secs_elapsed'], axis=0)
    total_time_ad = sessions.pivot_table(values='secs_elapsed', index='user_id', columns=['action_type_on_device'],
                                         aggfunc='sum')
    relative_ttad = total_time_ad.div(total_time['secs_elapsed'], axis=0)
    avg_time_ad = sessions.pivot_table(values='secs_elapsed', index='user_id', columns=['action_type_on_device'],
                                       aggfunc='mean')
    std_time_ad = sessions.pivot_table(values='secs_elapsed', index='user_id', columns=['action_type_on_device'],
                                       aggfunc='std')
    nunique_a_ad = sessions.pivot_table(values='action', index='user_id', columns=['action_type_on_device'],
                                        aggfunc=pd.Series.nunique)
    nunique_a_det_ad = sessions.pivot_table(values='action_detail', index='user_id', columns=['action_type_on_device'],
                                            aggfunc=pd.Series.nunique)

    users = users.merge(total_action, how='left', left_on='id', right_index=True)
    users = users.merge(total_actions_ad, how='left', left_on='id', right_index=True)
    users = users.merge(nunique_a, how='left', left_on='id', right_index=True)
    users = users.merge(nunique_a_type, how='left', left_on='id', right_index=True)
    users = users.merge(nunique_a_det, how='left', left_on='id', right_index=True)
    users = users.merge(nunique_device, how='left', left_on='id', right_index=True)
    users = users.merge(total_time, how='left', left_on='id', right_index=True)
    users = users.merge(total_time_ad, how='left', left_on='id', right_index=True)
    users = users.merge(avg_time_ad, how='left', left_on='id', right_index=True)
    users = users.merge(std_time_ad, how='left', left_on='id', right_index=True)
    users = users.merge(nunique_a_ad, how='left', left_on='id', right_index=True)
    users = users.merge(nunique_a_det_ad, how='left', left_on='id', right_index=True)
    users = users.merge(relative_taad, how='left', left_on='id', right_index=True)
    users = users.merge(relative_ttad, how='left', left_on='id', right_index=True)

    in_session = sessions[['user_id', 'sessions']].drop_duplicates()
    users = users.merge(in_session, how='left', left_on='id', right_on='user_id').drop('user_id', axis=1)
    users.sessions.fillna(0, inplace=True)

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


def validate(X_train=None, y_train=None, X_test=None, y_test=None, target=None, model=None, parameters=None,
             model_name=None):
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


def featureSelection(X_train, y_train, threshold):
    model = RandomForestClassifier(n_estimators=1000, class_weight='balanced', verbose=2, random_state=42)
    model.fit(X_train, y_train)
    fi = model.feature_importances_
    fi_cumulative = pd.DataFrame(fi, columns=['feat_importance']).sort_values('feat_importance',
                                                                              ascending=False).cumsum()
    fi_index = np.array(fi_cumulative[fi_cumulative['feat_importance'] <= threshold].index)
    with open(os.path.join(MODELS_PATH,'fi.bin'), 'wb') as f:
        pickle.dump(fi_index, f)
        f.close()


def validateFitModel(X_train, y_train, X_test, y_test, cv=False, target=None):
    rs = RobustScaler(quantile_range=(0.1, 0.90))
    mms = MinMaxScaler()
    X_train_mms = mms.fit_transform(rs.fit_transform(X_train))
    ncr = EditedNearestNeighbours(n_neighbors=1, sampling_strategy=[7, 10], random_state=42,
                                  return_indices=True)
    _, _, indexes = ncr.fit_resample(X_train_mms, y_train)
    resampling_index = random.sample(range(len(indexes)), len(indexes))
    sampled_indexes = indexes[resampling_index]
    with open(os.path.join(MODELS_PATH,'sampled_dfs_%s.bin' % target), 'wb') as f:
        pickle.dump(sampled_indexes, f)
        f.close()

    model = XGBClassifier(verbosity=2, n_estimators=100, objective='multi:softprob', learning_rate=0.125,
                          min_child_weight=1, max_depth=13, gamma=0.6, max_delta_step=0, subsample=1,
                          colsample_bytree=0.9, reg_lambda=2, scale_pos_weight=0.05)
    if cv:
        param_grid = {'n_estimators': [10],
                      'objective': ['multi:softprob'],
                      'learning_rate': [0.125],
                      'min_child_weigth': [1],
                      'max_depth': [13],
                      'gamma': [0.6],
                      'max_delta_step': [0],
                      'subsample': [1],
                      'colsample_bytree': [0.9],
                      'reg_lambda': [2],
                      'scale_pos_weight': [0.05]
                      }
        validate(X_train[sampled_indexes], y_train[sampled_indexes], X_test, y_test, target=target,
                 model=model, parameters=param_grid, model_name='XGB')
    else:
        model.fit(X_train[sampled_indexes], y_train[sampled_indexes])
        with open(os.path.join(MODELS_PATH,'%s_fitted_classifier.bin' % target), 'wb') as f:
            pickle.dump(model, f)
            f.close()

    return


if __name__ == '__main__':
    # users = pd.read_csv(os.path.join(os.getcwd(), 'data', 'train_users.csv'))

    # ''' AGE FIXING '''
    # users.replace('-unknown-', np.nan, inplace=True)
    # age = np.array(users.age)
    # np.place(age, age >= 110, np.nan)
    # np.place(age, age <= 14, np.nan)
    # np.place(age, age >= 80, 80)
    # users.loc[:, 'age'] = age
    # users.age.fillna(users.age.median(), inplace=True)

    # ''' DATETIME ENGINEERING '''
    # users.loc[:, 'timestamp_first_active'] = pd.to_datetime(users['timestamp_first_active'].astype(str).str[:8])
    # users.loc[:, 'date_account_created'] = pd.to_datetime(users['date_account_created'])
    # users.insert(1, 'time_gap_after_creation', (users.timestamp_first_active - users.date_account_created).dt.days)
    # users.insert(1, 'week_created', users.loc[:, 'date_account_created'].dt.week)
    # users.insert(1, 'week_first_active', users.loc[:, 'timestamp_first_active'].dt.week)
    # users.drop(['date_account_created', 'timestamp_first_active', 'date_first_booking'], axis=1, inplace=True)

    # ''' SESSIONS STATISTICS '''
    # users_session = elaborateOnlineActivityStatistics(users)

    # ''' CREATE TRAIN/TEST DATASETS  '''
    # users_session.insert(0, 'NANs', users_session.isnull().sum(axis=1))
    # Y = np.array(users_session['country_destination'])
    # train_df, test_df, _, _ = train_test_split(users_session, Y, test_size=0.10, random_state=42)
    # ohe, train_categorical_cols_encoded = categoricalTransformation(train_df, ohe=None)
    # train_df = pd.concat(
    #     (train_df[
    #          ['id', 'sessions', 'country_destination', 'time_gap_after_creation', 'week_created', 'week_first_active',
    #           'age', 'signup_flow', 'NANs']],
    #      train_categorical_cols_encoded, train_df.iloc[:, 17:-1].fillna(0)), axis=1).fillna(0)
    # _, test_categorical_cols_encoded = categoricalTransformation(test_df, ohe=ohe)
    # test_df = pd.concat(
    #     (test_df[
    #          ['id', 'sessions', 'country_destination', 'time_gap_after_creation', 'week_created', 'week_first_active',
    #           'age', 'signup_flow', 'NANs']],
    #      test_categorical_cols_encoded, test_df.iloc[:, 17:-1].fillna(0)),
    #     axis=1).fillna(0)

    '''SAVE TRAIN/TEST DF'''
    # with open('train_all', 'wb') as f:
    #     pickle.dump(train_df, f)
    #     f.close()
    # with open('test_all', 'wb') as f:
    #     pickle.dump(test_df, f)
    #     f.close()

    '''LOAD TRAIN/TEST DF'''
    with open('train_all', 'rb') as f:
        train_df = pickle.load(f)
        y_train = np.array(train_df['country_destination'])
        X_train = train_df.iloc[:, 3:]
        f.close()
    with open('test_all', 'rb') as f:
        test_df = pickle.load(f)
        y_test = np.array(test_df['country_destination'])
        X_test = test_df.iloc[:, 3:]
        f.close()
    test_session_mask = np.array(test_df[['sessions']] == 1).reshape(-1, )
    train_session_mask = np.array(train_df[['sessions']] == 1).reshape(-1)
    no_sessions_cols_ind = range(0, 91)  # relativo a X_train
    sessions_cols_ind = range(3, 447)  # relativo a X_train
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    ''' Split dataset in Sessions/NoSessions '''
    X_train_sessions = X_train[train_session_mask].iloc[:, sessions_cols_ind]
    y_train_sessions = y_train[train_session_mask]
    X_test_sessions = X_test[test_session_mask].iloc[:, sessions_cols_ind]
    y_test_sessions = y_test[test_session_mask]
    X_train_no_sessions = X_train.iloc[:, no_sessions_cols_ind]
    y_train_no_sessions = y_train
    X_test_no_sessions = X_test.iloc[:, no_sessions_cols_ind]
    y_test_no_sessions = y_test
    vt = VarianceThreshold()
    X_train_sessions = vt.fit_transform(X_train_sessions)
    X_test_sessions = vt.transform(X_test_sessions)
    X_train_no_sessions = vt.fit_transform(X_train_no_sessions)
    X_test_no_sessions = vt.transform(X_test_no_sessions)

    ''' Feature Selection (for sessions only)'''
    # featureSelection(X_train_sessions, y_train_sessions, 0.95)
    with open(os.path.join(MODELS_PATH,'fi.bin'), 'rb') as f:
        fi = pickle.load(f)
        f.close()
    X_train_sessions = X_train_sessions[:, fi].copy()
    X_test_sessions = X_test_sessions[:, fi].copy()

    '''Validation/fitting Models'''
    # validateFitModel(X_train_sessions, y_train_sessions, X_test_sessions, y_test_sessions, False, target='session')
    # validateFitModel(X_train_no_sessions, y_train_no_sessions, X_test_no_sessions, y_test_no_sessions, False,
    #                  target='no_session')

    ''' Prediction '''
    with open(os.path.join(MODELS_PATH,'session_fitted_classifier.bin'), 'rb') as f:
        sessions_model = pickle.load(f)
        f.close()

    with open(os.path.join(MODELS_PATH,'no_session_fitted_classifier.bin'), 'rb') as f:
        no_sessions_model = pickle.load(f)
        f.close()
    no_sessions_prediction = no_sessions_model.predict_proba(X_test_no_sessions)
    sessions_prediction = sessions_model.predict_proba(X_test_sessions)
    no_sessions_prediction[test_session_mask] = (no_sessions_prediction[test_session_mask] * 0.50) + (
            sessions_prediction * 0.50)
    weighted_prediction = np.argmax(no_sessions_prediction, axis=1)

    print('all test - best 10')
    print(confusion_matrix(y_test, weighted_prediction))
    print(balanced_accuracy_score(y_test, weighted_prediction))
    print(ndcg_score(y_test, no_sessions_prediction,4))

    print('sessions only - best 15')
    print(confusion_matrix(y_test[test_session_mask], weighted_prediction[test_session_mask]))
    print(balanced_accuracy_score(y_test[test_session_mask], weighted_prediction[test_session_mask]))
    print(ndcg_score(y_test[test_session_mask], sessions_prediction,4))
    print('p')

    df = pd.DataFrame(np.argsort(-no_sessions_prediction))
    df.insert(12,'true',y_test_no_sessions)
    dict = {}
    for i in range(12):
        dict[i] = le.inverse_transform([i])[0]
    df = df.applymap(lambda x: dict[x])
    print('p')
