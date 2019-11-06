import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def histogram(x_ticks, x1, x2, labelX1, labelX2, labelXAxis, labelYAxis, w, title, path):
    r1 = np.arange(len(x1))
    r2 = [x + w for x in r1]

    plt.bar(r1, x1, width=w, label=labelX1)
    plt.bar(r2, x2, width=w, label=labelX2)

    plt.title(title)
    plt.xlabel(labelXAxis, fontweight='bold')
    plt.ylabel(labelYAxis, fontweight='bold')
    len_max = max([len(str(tick)) for tick in x_ticks])
    if len_max > 3:
        rotation = 45
    else:
        rotation = 0
    plt.xticks([r + w for r in range(len(r1))], x_ticks, rotation=rotation)
    plt.legend()
    plt.savefig(path)
    plt.clf()
    plt.close()

    return


def multiLinePlot(x_ticks, Y_s, labels, title):
    fig = plt.figure()

    return


def ageBucketAnalysis():
    age_gender_bkts = pd.read_csv(os.path.join(os.getcwd(), 'data', 'age_gender_bkts.csv'))
    for age_bucket in np.unique(age_gender_bkts['age_bucket']):
        ab_filtered = age_gender_bkts[age_gender_bkts['age_bucket'] == age_bucket].sort_values('country_destination')
        males = ab_filtered[ab_filtered['gender'] == 'male']
        females = ab_filtered[ab_filtered['gender'] == 'female']
        x = sorted(np.unique(ab_filtered.country_destination))
        y_males = males['population_in_thousands']
        y_females = females['population_in_thousands']

        histogram(x, y_males, y_females, 'males', 'females', 'Country', 'Pop (*1000)', 0.3,
                  'Age bucket %s divided in genders and countries' % age_bucket,
                  os.path.join(os.getcwd(), 'exploratoryAnalysis', 'age_bucket_%s.png' % age_bucket))


def getModel(layers=None):
    model = Sequential()

    model.add(Dense(units=layers[1], activation='softmax', input_dim=132))
    for layer in layers[1:]:
        model.add(Dense(units=layer, activation='softmax'))
        model.add(Dropout(0, 25))

    model.add(Dense(units=12, activation='softmax'))

    model.compile(optimizer=Adam(), loss='mse')

    return model


def prediction(train_users):
    X = train_users[
        ['age', 'gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider',
         'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']]
    Y = train_users['country_destination']
    X = pd.get_dummies(X)
    Y = pd.get_dummies(Y)
    X['age'].fillna(X['age'].mode()[0], inplace=True)
    X['signup_flow'].fillna(X['signup_flow'].mode()[0], inplace=True)

    mms = MinMaxScaler()
    X = mms.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    # cl = RandomForestClassifier()

    cl = getModel([280, 560, 190])
    cl.fit(np.array(X_train), np.array(y_train), epochs=100, verbose=0)
    prediction = cl.predict(X_test)
    prediction_df = pd.DataFrame({'predicted': prediction, 'true': y_test})
    prediction_df.to_csv('test.csv')


def sessionsTrain(boolean_booked=0):
    sessions = pd.read_csv(os.path.join(os.getcwd(), 'data', 'sessions.csv'))
    train_users = pd.read_csv(os.path.join(os.getcwd(), 'data', 'train_users.csv'))
    if boolean_booked:
        train_users['booked'] = np.array(train_users['country_destination'] != 'NDF').astype(int)
    sessions_train = sessions[sessions['user_id'].isin(train_users['id'])]
    sessions_train['min_elapsed'] = sessions_train['secs_elapsed'] / 60
    sessions_train.drop(['secs_elapsed'], axis=1, inplace=True)
    sessions_train = sessions_train.merge(train_users[['id', ['country_destination', 'booked'][boolean_booked]]],
                                          left_on='user_id', right_on='id').drop('id', axis=1)

    return sessions_train


def sessionsAnalysisBooked(col_to_analyze='action_type', sessions_train=None):
    sessions_train['count'] = 1

    ''' Analisi durata e quantità azioni'''
    for j, val in enumerate(['min_elapsed', 'count']):
        pivot = sessions_train[['user_id', col_to_analyze, val, 'booked']].pivot_table(
            index=['user_id', col_to_analyze, 'booked'],
            values=val,
            aggfunc='sum').reset_index()
        x = sorted(np.unique(pivot[col_to_analyze]))
        pivot_0 = pivot[pivot['booked'] == 0]
        pivot_0_pivot = pivot_0.pivot_table(index=col_to_analyze, values=val, aggfunc='median')
        pivot_1 = pivot[pivot['booked'] == 1]
        pivot_1_pivot = pivot_1.pivot_table(index=col_to_analyze, values=val, aggfunc='median')
        histogram(x, pivot_0_pivot[val], pivot_1_pivot[val], 'Not booked', 'Booked', 'Action Type', '%s' % val, 0.3,
                  'Mediana %s tipologie di azioni' % (('del tempo totale speso dall\' utente sulle singole',
                                                       'delle azioni totali per utente per singole')[j]),
                  os.path.join(os.getcwd(), 'exploratoryAnalysis',
                               'Mediana %s tipologie di azioni' % (('del tempo totale speso dall\' utente sull singole',
                                                                    'delle azioni totali per utente per singole')[j])))

    ''' Analisi eterogeneità azioni'''
    tab1 = sessions_train[sessions_train['booked'] == 0].pivot_table(index='user_id', values='action_type',
                                                                     aggfunc=lambda x: len(np.unique(x.astype(str))))
    tab2 = sessions_train[sessions_train['booked'] == 1].pivot_table(index='user_id', values='action_type',
                                                                     aggfunc=lambda x: len(np.unique(x.astype(str))))
    x1 = np.array(np.unique(tab1['action_type'], return_counts=True)).astype(float)
    x1[1] = x1[1] / x1[1].sum()
    x2 = np.array(np.unique(tab2['action_type'], return_counts=True)).astype(float)
    x2[1] = x2[1] / x2[1].sum()

    histogram(x1[0], x1[1], x2[1], 'Not Booked', 'Booked', 'N° different actions', 'relative frequency', 0.3,
              'Booked/Not booked analysis of Action heterogeneity', os.path.join(os.getcwd(), 'exploratoryAnalysis',
                                                                                 'Heterogeneity of actions in booked vs not booked.png'))

    return


def elaborateOnlineActivityStatistics(train_users=None):
    sessions = pd.read_csv(os.path.join(os.getcwd(), 'data', 'sessions.csv'))
    sessions_train = sessions[sessions['user_id'].isin(train_users['id'])]
    del sessions
    sessions_train['min_elapsed'] = sessions_train['secs_elapsed'] / 60
    sessions_train.drop(['secs_elapsed'], axis=1, inplace=True)

    pivot_sum_time = sessions_train.pivot_table(values='min_elapsed', index='user_id', columns='action_type',
                                                aggfunc='sum').fillna(0)
    pivot_sum_time.columns = [x + '_total_time' for x in pivot_sum_time.columns]
    pivot_sum_count = sessions_train.pivot_table(values='min_elapsed', index='user_id', columns='action_type',
                                                 aggfunc='count').fillna(0)
    pivot_sum_count.columns = [x + '_total_events' for x in pivot_sum_time.columns]
    pivot_heterogeneity = sessions_train.pivot_table(values='action_type', index='user_id',
                                                     aggfunc=lambda x: len(np.unique(x.astype(str))))
    pivot_heterogeneity.columns = ['heterogeneity']

    train_users = train_users.merge(pivot_sum_time, how='inner', left_on='id', right_index=True)
    train_users = train_users.merge(pivot_sum_count, how='inner', left_on='id', right_index=True)
    train_users = train_users.merge(pivot_heterogeneity, how='inner', left_on='id', right_index=True)

    return train_users


def sessionsAnalysisDestination(col_to_analyze='action_type', sessions_train=None):
    sessions_train['count'] = 1
    values = {}

    ''' Analisi durata e quantità '''
    for action in sorted(sessions_train[col_to_analyze].astype(str).unique()):
        values[action] = 0
    for j, val in enumerate(['min_elapsed', 'count']):
        for country in sessions_train['country_destination'].unique():
            country_values = values.copy()
            df = sessions_train[sessions_train['country_destination'] == country]
            pivot = df[['user_id', col_to_analyze, 'min_elapsed', 'count']].pivot_table(
                index=['user_id', col_to_analyze],
                values=val,
                aggfunc='sum').reset_index()
            pivot_ = pivot.pivot_table(index=col_to_analyze, values=val, aggfunc='median')
            for i in pivot_.index:
                country_values[i] = pivot_.loc[i][0]
            plt.plot(pd.Series(country_values).sort_index().values, linestyle='--', marker='_', label=country)

        xticks = sorted(sessions_train[col_to_analyze].astype(str).unique())
        plt.xticks(range(len(xticks)), xticks, rotation=45)
        plt.title(['Tempo totale speso sulle singole azioni, diviso per Paese di destinazione',
                   'Numero totale azioni per tipologia, diviso per Paese di destinazione'][j])
        plt.xlabel(col_to_analyze, fontweight='bold')
        plt.legend()
        plt.ylabel(['Median tempo speso', 'Mediana numero azioni'][j], fontweight='bold')
        plt.savefig(os.path.join(os.getcwd(), 'exploratoryAnalysis',
                                 ['Analisi tempo totale impegato sulle singole azioni per Paese di destinazione.png',
                                  'Numero totale azioni per tipologia per Paese di destinazione.png'][j]))
        plt.clf()
        plt.close()

    ''' Analisi Eterogeneità '''
    for country in sessions_train['country_destination'].unique():
        df = sessions_train[sessions_train['country_destination'] == country]
        tab = df.pivot_table(index='user_id', values=col_to_analyze, aggfunc=lambda x: len(np.unique(x.astype(str))))
        x = np.array(np.unique(tab['action_type'], return_counts=True)).astype(float)
        x[1] = x[1] / x[1].sum()
        plt.plot(x[0], x[1], linestyle='--', marker='_', label=country)
    plt.xlabel(col_to_analyze, fontweight='bold')
    plt.ylabel('Frequenza relativa', fontweight='bold')
    plt.title('Eterogeneità azioni paese per paese')
    plt.legend()
    plt.savefig(os.path.join(os.getcwd(), 'exploratoryAnalysis', 'Analisi eterogeneità azioni per singolo paese.png'))
    plt.clf()
    plt.close()
    return


if __name__ == '__main__':
    # ageBucketAnalysis()
    # sessionsAnalysisBooked('action_type', sessionsTrain(boolean_booked=1))
    # sessionsAnalysisDestination('action_type',sessionsTrain(boolean_booked=0))
    train_users = pd.read_csv(os.path.join(os.getcwd(), 'data', 'train_users.csv'))
    elaborateOnlineActivityStatistics(train_users)
    # age_gender_bkts = pd.read_csv(os.path.join(os.getcwd(), 'data', 'age_gender_bkts.csv'))
    # countries = pd.read_csv(os.path.join(os.getcwd(), 'data', 'countries.csv'))

    # train_users = pd.read_csv(os.path.join(os.getcwd(), 'data', 'train_users.csv'))
    # prediction(train_users)
    #
    #

    '''session analysis of who booked'''
    # mask_who_booked_in2014 = train_users['country_destination'] != 'NDF'
    # who_booked_after2014 = train_users[mask_who_booked_in2014]['id']
    # sessions_who_booked = sessions_train[sessions_train['user_id'].isin(who_booked_after2014)]
    # n_booked_with_online_data = sessions_who_booked['user_id'].nunique()
    #
    # df_actions_counts = pd.DataFrame({'actions': np.array(
    #     np.unique(np.array(sessions_who_booked['action'].astype(str)), return_counts=True))[0], 'count':
    #                                       np.array(np.unique(np.array(sessions_who_booked['action'].astype(str)),
    #                                                          return_counts=True))[1]})
    # df_actionsType_counts = pd.DataFrame({'actions': np.array(
    #     np.unique(np.array(sessions_who_booked['action_type'].astype(str)), return_counts=True))[0], 'count':
    #                                       np.array(np.unique(np.array(sessions_who_booked['action_type'].astype(str)),
    #                                                          return_counts=True))[1]})
    # df_actionsDetail_counts = pd.DataFrame({'actions': np.array(
    #     np.unique(np.array(sessions_who_booked['action_detail'].astype(str)), return_counts=True))[0], 'count':
    #                                       np.array(np.unique(np.array(sessions_who_booked['action_detail'].astype(str)),
    #                                                          return_counts=True))[1]})
    #
    #
    #
    # '''session analysis of who didnt book'''
    # mask_who_didntbook_in2014 = train_users['country_destination'] == 'NDF'
    # who_didntbook_after2014 = train_users[mask_who_didntbook_in2014]['id']
    # sessions_who_didntbook = sessions_train[sessions_train['user_id'].isin(who_didntbook_after2014)]
    # n_didntbook_with_online_data = sessions_who_didntbook['user_id'].nunique()
    #
    # didntbook_df_actions_counts = pd.DataFrame({'actions': np.array(
    #     np.unique(np.array(sessions_who_didntbook['action'].astype(str)), return_counts=True))[0], 'count':
    #                                       np.array(np.unique(np.array(sessions_who_didntbook['action'].astype(str)),
    #                                                          return_counts=True))[1]})
    # didntbook_df_actionsType_counts = pd.DataFrame({'actions': np.array(
    #     np.unique(np.array(sessions_who_didntbook['action_type'].astype(str)), return_counts=True))[0], 'count':
    #                                           np.array(
    #                                               np.unique(np.array(sessions_who_didntbook['action_type'].astype(str)),
    #                                                         return_counts=True))[1]})
    # didntbook_df_actionsDetail_counts = pd.DataFrame({'actions': np.array(
    #     np.unique(np.array(sessions_who_didntbook['action_detail'].astype(str)), return_counts=True))[0], 'count':
    #                                             np.array(np.unique(np.array(sessions_who_didntbook['action_detail'].astype(str)),
    #                                                          return_counts=True))[1]})
    #

    #
    #
    # test_users = pd.read_csv(os.path.join(os.getcwd(), 'data', 'test_users.csv'))
    # # countries.describe()
