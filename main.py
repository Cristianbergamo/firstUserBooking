import os

import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def getModel(layers=None, dropout_rate=None, optimizer=None, regularizer=None, loss_function=None, output_units=2):
    model = Sequential()
    model.add(Dense(units=layers[1], activation='relu', input_dim=33, kernel_regularizer=regularizer))
    if dropout_rate is not None:
        model.add(Dropout(dropout_rate))

    for layer in layers[1:]:
        model.add(Dense(units=layer, activation='relu', kernel_regularizer=regularizer))
        if dropout_rate is not None:
            model.add(Dropout(dropout_rate))

    model.add(Dense(units=output_units, activation='softmax'))

    model.compile(optimizer=optimizer, loss=loss_function, metrics=['acc'])

    return model


def predict(train_users, layers=[], optimizer=None, loss_function=None, output_units=None):
    Y = train_users['country_destination']
    X = train_users.iloc[:, 2:]
    Y_one_hot = pd.get_dummies(Y)

    mms = MinMaxScaler()

    X_train, X_test, y_train, y_test = train_test_split(X, Y_one_hot, test_size=0.33, random_state=42)

    X_train = mms.fit_transform(X_train)
    model = getModel(layers=layers, optimizer=optimizer, regularizer=None, dropout_rate=None,
                     loss_function=loss_function, output_units=output_units)
    count_countries = np.unique(Y, return_counts=True)
    max_count = np.argmax(count_countries[1])
    class_weight = {}
    for j, i in enumerate(count_countries[1]):
        class_weight[j] = count_countries[1][max_count] / i
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    #
    # '''oversampling minority classes'''
    # sm = SMOTE()
    # X_train, y_train = sm.fit_sample(np.array(X_train), np.array(y_train))

    model.fit(X_train, y_train, epochs=100, class_weight=class_weight, batch_size=32)

    prediction = model.predict(mms.transform(X_test))
    prediction = np.argmax(prediction, axis=1)
    y_test = np.argmax(y_test, axis=1)
    print('accuracy: %s', str(accuracy_score(y_true=y_test, y_pred=prediction)))

    return prediction
    # prediction_df = pd.DataFrame({'predicted': prediction, 'true': y_test})
    # prediction_df.to_csv('test.csv')


def multiModelPrediction(users):
    NDF_df = users.copy()
    NDF_df['country_destination'] = np.array(NDF_df['country_destination'] == 'NDF').astype(int)
    prediction_NDF = predict(NDF_df, layers=[8, 8, 8], optimizer=SGD(lr=0.0005), loss_function='binary_crossentropy',
                             output_units=2)
    NDF_df['prediction'] = prediction_NDF

    USA_df = NDF_df[prediction_NDF == 0]
    USA_df['country_destination'] = np.array(USA_df['country_destination'] == 'US').astype(int)
    prediction_USA = predict(USA_df, layers=[8, 8, 8], loss_function='binary_crossentropy', output_units=2)
    USA_df['prediction'] = prediction_USA

    countries_df = USA_df[prediction_USA == 0]
    prediction_countries = predict(countries_df, layers=[12, 12, 12, 12, 12], loss_function='categorical_crossentropy',
                                   output_units=10)
    countries_df['prediction'] = prediction_countries

    result = pd.concat((NDF_df[['id', 'prediction']], USA_df[['id', 'prediction']], countries_df[['id', 'prediction']]),
                       axis=0)

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
    pca = PCA(14)
    categoricalToPca = pd.DataFrame(pca.fit_transform(dummy_users))

    return categoricalToPca


if __name__ == '__main__':
    train_users = pd.read_csv(os.path.join(os.getcwd(), 'data', 'train_users.csv'))
    train_users.drop('age', axis=1, inplace=True)
    categorical_cols_encoded = categoricalToPca(train_users)
    users = pd.concat((train_users[['id', 'country_destination']], categorical_cols_encoded), axis=1)

    full_df = elaborateOnlineActivityStatistics(users)
    # predicted = predict(full_df,2)
    multiModelPrediction(users)
