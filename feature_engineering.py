import pandas as pd


def generate_event_features(users_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    df: pd.DataFrame = users_df.copy()
    events_df = events_df.copy()

    events_df = pd.merge(left=events_df, right=df[['reg_ts']], left_on='user_id', right_index=True, how='left')

    events_df['d0'] = events_df['reg_ts'].dt.normalize() + pd.Timedelta(days=0, hours=23, minutes=59, seconds=59)
    events_df['d1'] = events_df['reg_ts'].dt.normalize() + pd.Timedelta(days=1, hours=23, minutes=59, seconds=59)
    events_df['d3'] = events_df['reg_ts'].dt.normalize() + pd.Timedelta(days=3, hours=23, minutes=59, seconds=59)
    events_df['d7'] = events_df['reg_ts'].dt.normalize() + pd.Timedelta(days=7, hours=23, minutes=59, seconds=59)

    battle_features = __generate_battle_features(df, events_df)
    session_features = __generate_session_features(df, events_df)
    wealth_features = __generate_wealth_features(df, events_df)
    quest_features = __generate_quest_features(df, events_df)
    level_features = __generate_level_features(df, events_df)
    payment_features = __generate_payment_features(df, events_df)

    return df[[]] \
        .join(battle_features, how='left') \
        .join(session_features, how='left') \
        .join(wealth_features, how='left') \
        .join(quest_features, how='left') \
        .join(level_features, how='left') \
        .join(payment_features, how='left')


def __generate_battle_features(users_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    df = users_df.copy()
    b_df = events_df[events_df['event_name'] == 'battle']

    b_df_1 = b_df[b_df['event_value'] == 1]
    b_df_0 = b_df[b_df['event_value'] == 0]

    df['battles_won_d0'] = b_df_1[b_df_1['event_ts'] <= b_df_1['d0']]['user_id'].value_counts()
    df['battles_won_d1'] = b_df_1[b_df_1['event_ts'] <= b_df_1['d1']]['user_id'].value_counts()
    df['battles_won_d3'] = b_df_1[b_df_1['event_ts'] <= b_df_1['d3']]['user_id'].value_counts()
    df['battles_won_d7'] = b_df_1[b_df_1['event_ts'] <= b_df_1['d7']]['user_id'].value_counts()

    df['battles_lost_d0'] = b_df_0[b_df_0['event_ts'] <= b_df_0['d0']]['user_id'].value_counts()
    df['battles_lost_d1'] = b_df_0[b_df_0['event_ts'] <= b_df_0['d1']]['user_id'].value_counts()
    df['battles_lost_d3'] = b_df_0[b_df_0['event_ts'] <= b_df_0['d3']]['user_id'].value_counts()
    df['battles_lost_d7'] = b_df_0[b_df_0['event_ts'] <= b_df_0['d7']]['user_id'].value_counts()

    df['battles_won_d0'] = df['battles_won_d0'].fillna(0)
    df['battles_won_d1'] = df['battles_won_d1'].fillna(0)
    df['battles_won_d3'] = df['battles_won_d3'].fillna(0)
    df['battles_won_d7'] = df['battles_won_d7'].fillna(0)

    df['battles_lost_d0'] = df['battles_lost_d0'].fillna(0)
    df['battles_lost_d1'] = df['battles_lost_d1'].fillna(0)
    df['battles_lost_d3'] = df['battles_lost_d3'].fillna(0)
    df['battles_lost_d7'] = df['battles_lost_d7'].fillna(0)

    return df[[
        'battles_won_d0', 'battles_won_d1', 'battles_won_d3', 'battles_won_d7',
        'battles_lost_d0', 'battles_lost_d1', 'battles_lost_d3', 'battles_lost_d7',
    ]]


def __generate_session_features(users_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    df = users_df.copy()
    l_df = events_df[events_df['event_name'] == 'login'].copy()

    l_df['event_date'] = l_df['event_ts'].dt.normalize()

    l_df_d0 = l_df[l_df['event_ts'] <= l_df['d0']]
    l_df_d1 = l_df[l_df['event_ts'] <= l_df['d1']]
    l_df_d3 = l_df[l_df['event_ts'] <= l_df['d3']]
    l_df_d7 = l_df[l_df['event_ts'] <= l_df['d7']]

    df['session_time_d0'] = l_df_d0[['user_id', 'event_value']].groupby(by='user_id').sum()['event_value']
    df['session_time_d1'] = l_df_d1[['user_id', 'event_value']].groupby(by='user_id').sum()['event_value']
    df['session_time_d3'] = l_df_d3[['user_id', 'event_value']].groupby(by='user_id').sum()['event_value']
    df['session_time_d7'] = l_df_d7[['user_id', 'event_value']].groupby(by='user_id').sum()['event_value']

    df['session_time_d0'] = df['session_time_d0'].fillna(0)
    df['session_time_d1'] = df['session_time_d1'].fillna(0)
    df['session_time_d3'] = df['session_time_d3'].fillna(0)
    df['session_time_d7'] = df['session_time_d7'].fillna(0)

    df['inactive_d1'] = (df['session_time_d1'] == df['session_time_d0']).astype(int)

    df['n_active_days'] = l_df[['user_id', 'event_date']].groupby(by='user_id').nunique()['event_date']

    df['n_active_days'] = df['n_active_days'].fillna(0).astype(int)

    return df[[
        'session_time_d0', 'session_time_d1', 'session_time_d3', 'session_time_d7',
        'inactive_d1', 'n_active_days',
    ]]


def __generate_wealth_features(users_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    df = users_df.copy()
    w_df = events_df[events_df['event_name'] == 'wealth_on_login']

    w_df_d0 = w_df[w_df['event_ts'] <= w_df['d0']]
    w_df_d1 = w_df[w_df['event_ts'] <= w_df['d1']]
    w_df_d3 = w_df[w_df['event_ts'] <= w_df['d3']]
    w_df_d7 = w_df[w_df['event_ts'] <= w_df['d7']]

    df['wealth_on_login_max_d0'] = w_df_d0[['user_id', 'event_value']].groupby(by='user_id').max()['event_value']
    df['wealth_on_login_max_d1'] = w_df_d1[['user_id', 'event_value']].groupby(by='user_id').max()['event_value']
    df['wealth_on_login_max_d3'] = w_df_d3[['user_id', 'event_value']].groupby(by='user_id').max()['event_value']
    df['wealth_on_login_max_d7'] = w_df_d7[['user_id', 'event_value']].groupby(by='user_id').max()['event_value']

    df['wealth_on_login_max_d0'] = df['wealth_on_login_max_d0'].fillna(0)
    df['wealth_on_login_max_d1'] = df['wealth_on_login_max_d1'].fillna(0)
    df['wealth_on_login_max_d3'] = df['wealth_on_login_max_d3'].fillna(0)
    df['wealth_on_login_max_d7'] = df['wealth_on_login_max_d7'].fillna(0)

    df['wealth_on_login_max_d0=802'] = (df['wealth_on_login_max_d0'] == 802).astype(int)
    df['wealth_on_login_max_d7=802'] = (df['wealth_on_login_max_d7'] == 802).astype(int)

    return df[[
        'wealth_on_login_max_d0', 'wealth_on_login_max_d1', 'wealth_on_login_max_d3', 'wealth_on_login_max_d7',
        'wealth_on_login_max_d0=802', 'wealth_on_login_max_d7=802',
    ]]


def __generate_quest_features(users_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    df = users_df.copy()
    f_df = events_df[events_df['event_name'] == 'finish_quest']

    f_df_d0 = f_df[f_df['event_ts'] <= f_df['d0']]
    f_df_d1 = f_df[f_df['event_ts'] <= f_df['d1']]
    f_df_d3 = f_df[f_df['event_ts'] <= f_df['d3']]
    f_df_d7 = f_df[f_df['event_ts'] <= f_df['d7']]

    df['finish_quest_sum_d0'] = f_df_d0[['user_id', 'event_value']].groupby(by='user_id').sum()['event_value']
    df['finish_quest_sum_d1'] = f_df_d1[['user_id', 'event_value']].groupby(by='user_id').sum()['event_value']
    df['finish_quest_sum_d3'] = f_df_d3[['user_id', 'event_value']].groupby(by='user_id').sum()['event_value']
    df['finish_quest_sum_d7'] = f_df_d7[['user_id', 'event_value']].groupby(by='user_id').sum()['event_value']

    df['finish_quest_sum_d0'] = df['finish_quest_sum_d0'].fillna(0).astype(int)
    df['finish_quest_sum_d1'] = df['finish_quest_sum_d1'].fillna(0).astype(int)
    df['finish_quest_sum_d3'] = df['finish_quest_sum_d3'].fillna(0).astype(int)
    df['finish_quest_sum_d7'] = df['finish_quest_sum_d7'].fillna(0).astype(int)

    return df[[
        'finish_quest_sum_d0', 'finish_quest_sum_d1', 'finish_quest_sum_d3', 'finish_quest_sum_d7',
    ]]


def __generate_level_features(users_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    df = users_df.copy()
    lv_df = events_df[events_df['event_name'] == 'level_up']

    lv_df_d0 = lv_df[lv_df['event_ts'] <= lv_df['d0']]
    lv_df_d1 = lv_df[lv_df['event_ts'] <= lv_df['d1']]
    lv_df_d3 = lv_df[lv_df['event_ts'] <= lv_df['d3']]
    lv_df_d7 = lv_df[lv_df['event_ts'] <= lv_df['d7']]

    df['level_up_max_d0'] = lv_df_d0[['user_id', 'event_value']].groupby(by='user_id').max()['event_value']
    df['level_up_max_d1'] = lv_df_d1[['user_id', 'event_value']].groupby(by='user_id').max()['event_value']
    df['level_up_max_d3'] = lv_df_d3[['user_id', 'event_value']].groupby(by='user_id').max()['event_value']
    df['level_up_max_d7'] = lv_df_d7[['user_id', 'event_value']].groupby(by='user_id').max()['event_value']

    df['level_up_max_d0'] = df['level_up_max_d0'].fillna(0).astype(int)
    df['level_up_max_d1'] = df['level_up_max_d1'].fillna(0).astype(int)
    df['level_up_max_d3'] = df['level_up_max_d3'].fillna(0).astype(int)
    df['level_up_max_d7'] = df['level_up_max_d7'].fillna(0).astype(int)

    return df[[
        'level_up_max_d0', 'level_up_max_d1', 'level_up_max_d3', 'level_up_max_d7',
    ]]


def __generate_payment_features(users_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    df = users_df.copy()
    p_df = events_df[events_df['event_name'] == 'payment']

    p_df_d0 = p_df[p_df['event_ts'] <= p_df['d0']]
    p_df_d1 = p_df[p_df['event_ts'] <= p_df['d1']]
    p_df_d3 = p_df[p_df['event_ts'] <= p_df['d3']]
    p_df_d7 = p_df[p_df['event_ts'] <= p_df['d7']]

    df['payment_sum_d0'] = p_df_d0[['user_id', 'event_value']].groupby(by='user_id').sum()['event_value']
    df['payment_sum_d1'] = p_df_d1[['user_id', 'event_value']].groupby(by='user_id').sum()['event_value']
    df['payment_sum_d3'] = p_df_d3[['user_id', 'event_value']].groupby(by='user_id').sum()['event_value']
    df['payment_sum_d7'] = p_df_d7[['user_id', 'event_value']].groupby(by='user_id').sum()['event_value']

    df['payment_sum_d0'] = df['payment_sum_d0'].fillna(0)
    df['payment_sum_d1'] = df['payment_sum_d1'].fillna(0)
    df['payment_sum_d3'] = df['payment_sum_d3'].fillna(0)
    df['payment_sum_d7'] = df['payment_sum_d7'].fillna(0)

    return df[[
        'payment_sum_d0', 'payment_sum_d1', 'payment_sum_d3', 'payment_sum_d7',
    ]]
