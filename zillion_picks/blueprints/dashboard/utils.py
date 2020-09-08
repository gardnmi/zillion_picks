import pandas as pd
import numpy as np
from pathlib import Path


def create_sidebar(path):

    # Create Side Bar Links
    df = pd.DataFrame([(file.stem, file.parent.stem) for file in path.rglob(
        '*.csv')], columns=['Season_Week', 'is_premium'])
    df[['Season', 'Week']] = df.Season_Week.str.split('_', n=2, expand=True)
    df['is_premium'] = df.is_premium.str.capitalize()

    df = df.sort_values(by=['Season', 'Week'], ascending=False)

    group = df.groupby('Season')

    links = {}

    for key, frame in group:
        links[key] = list(
            frame[['Week', 'is_premium']].itertuples(index=False))

    return links


def table_cleanup(df, week):

    cols = ['start_date', 'home_team', 'away_team', 'pre_game_spread', 'predicted_spread', 'Spread Difference', 'actual_spread',
            'spread_pick', 'spread_result', 'straight_pick', 'straight_result', 'home_conference', 'away_conference', 'season', 'week']

    col_names = {
        'season': 'Season',
        'week': 'Week',
        'start_date': 'Date',
        'home_team': 'Home Team',
        'home_conference': 'Home Conference',
        'away_team': 'Away Team',
        'away_conference': 'Away Conference',
        'pre_game_spread': 'Vegas Spread',
        'predicted_spread': 'Predicted Outcome',
        'spread_pick': 'Spread Pick',
        'straight_pick': 'Straight Up Pick',
        'actual_spread': 'Actual Spread',
        'spread_result': 'Spread Result',
        'straight_result': 'Straight Up Result'}

    df['Spread Difference'] = np.abs(
        df['predicted_spread'] - df['pre_game_spread'])

    if week == 'postseason':
        df['week'] = 'Post Season'
    else:
        pass

    df['start_date'] = pd.to_datetime(df['start_date']).dt.date

    df = df[cols]
    df = df.rename(columns=col_names)

    df = df.fillna('TBD')

    return df


def get_results(df, filepath, season, week):
    # Current Week Results
    try:
        spread_dict = df['spread_result'].value_counts().to_dict()
        straight_dict = df['straight_result'].value_counts().to_dict()

        spread_results = {}
        for key, value in spread_dict.items():
            spread_results[key] = value / sum(spread_dict.values())

        straight_results = {}
        for key, value in straight_dict.items():
            straight_results[key] = value / sum(straight_dict.values())
    except:
        spread_results = {}
        straight_results = {}

    try:
        # Season To Date Results
        files_df = pd.DataFrame([(file.stem, file) for file in filepath.rglob(
            './*.csv')], columns=['Season_Week', 'file_path'])
        files_df[['Season', 'Week']] = files_df.Season_Week.str.split(
            '_', n=2, expand=True)
        files_df = files_df.replace('postseason', np.nan)
        files_df['Week Nums'] = files_df.groupby('Season').transform(
            lambda x: x.reset_index().index+1)['Week']

        if week == '01' and len(spread_results) == 0:
            files_df = files_df[files_df['Season'].astype(
                int) == (int(season) - 1)]

        elif week == 'postseason' and len(spread_results) > 0:
            files_df = files_df[files_df['Season'].astype(
                int) == df['season'].unique()[0]]

        elif week == 'postseason' and len(spread_results) == 0:
            files_df = files_df[(files_df['Week Nums'].isnull()) & (
                files_df['Season'].astype(int) == df['season'].unique()[0])]

        elif len(spread_results) > 0:
            files_df = files_df[(files_df['Week Nums'].map(lambda x: int(x)) <= df['week'].unique()[
                                 0]) & (files_df['Season'].astype(int) == df['season'].unique()[0])]

        else:
            files_df = files_df[(files_df['Week Nums'].map(lambda x: int(x)) < df['week'].unique()[
                                 0]) & (files_df['Season'].astype(int) == df['season'].unique()[0])]

        dfs = []
        for file_name in files_df.file_path:
            df = pd.read_csv(file_name)
            dfs.append(df)

        all_df = pd.concat(dfs)

        std_spread_dict = all_df['spread_result'].value_counts().to_dict()
        std_straight_dict = all_df['straight_result'].value_counts().to_dict()

        std_spread_results = {}
        for key, value in std_spread_dict.items():
            std_spread_results[key] = value / sum(std_spread_dict.values())

        std_straight_results = {}
        for key, value in std_straight_dict.items():
            std_straight_results[key] = value / sum(std_straight_dict.values())
    except:
        std_spread_results = {}
        std_straight_results = {}

    return spread_results, straight_results, std_spread_results, std_straight_results
