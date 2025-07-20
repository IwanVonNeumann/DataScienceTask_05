import pandas as pd


def generate_features(users_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    df = users_df.copy()
    events_df = events_df.copy()

    events_df = pd.merge(left=events_df, right=df[['reg_ts']], left_on='user_id', right_index=True, how='left')

    events_df['d0'] = events_df['reg_ts'].dt.normalize() + pd.Timedelta(days=0, hours=23, minutes=59, seconds=59)
    events_df['d1'] = events_df['reg_ts'].dt.normalize() + pd.Timedelta(days=1, hours=23, minutes=59, seconds=59)
    events_df['d3'] = events_df['reg_ts'].dt.normalize() + pd.Timedelta(days=3, hours=23, minutes=59, seconds=59)
    events_df['d7'] = events_df['reg_ts'].dt.normalize() + pd.Timedelta(days=7, hours=23, minutes=59, seconds=59)

    battle_features = __generate_battle_features(df, events_df)

    return df


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
