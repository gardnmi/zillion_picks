import ast
import re
import urllib
from pathlib import Path
import numpy as np
import pandas as pd


def game_stat_transform(filepath=Path('data')):

    dfs = []
    for file in filepath.rglob('games_stats.csv'):

        df = pd.read_csv(file)

        split_columns = ['thirdDownEffAway', 'thirdDownEffHome',
                         'fourthDownEffAway', 'fourthDownEffHome',
                         'completionAttemptsAway', 'completionAttemptsHome']

        for column in split_columns:

            success_column = f'{column[:-4]}Success{column[-4:]}'
            attempt_column = f'{column[:-4]}Attempts{column[-4:]}'

            df[[success_column, attempt_column]] = df[column].str.split(
                '-', expand=True).iloc[:, :2].astype(float)
            df[column] = df[success_column] / df[attempt_column]

        split_columns = ['totalPenaltiesYardsAway', 'totalPenaltiesYardsHome']

        for column in split_columns:

            new_column = re.sub('Yards', '', column)

            df[[new_column, column]] = df[column].str.split(
                '-', n=1, expand=True).astype(float)

        time_columns = ['possessionTimeAway', 'possessionTimeHome']

        for column in time_columns:

            df[column] = pd.to_timedelta(df[column] + ':00').dt.total_seconds()

        df = df.fillna(0)

        df = df.drop(['home', 'away'], axis=1)

        dfs.append(df)

    df = pd.concat(dfs, axis=0, ignore_index=True)
    return df


def games_advanced_stats_transform(filepath=Path('data')):

    dfs = []

    for files in zip(filepath.rglob('games_advanced_stats.csv'), filepath.rglob('games.csv')):

        df = pd.read_csv(files[0])
        games_df = pd.read_csv(files[1])

        df = df.drop(['week', 'opponent', 'season'], axis=1)

        home_df = df.merge(games_df[['id', 'home_team']], left_on=['gameId', 'team'], right_on=[
                           'id', 'home_team'], how='inner').drop(['id', 'home_team'], axis=1)
        away_df = df.merge(games_df[['id', 'away_team']], left_on=['gameId', 'team'], right_on=[
                           'id', 'away_team'], how='inner').drop(['id', 'away_team'], axis=1)

        columns = home_df.columns.difference(['gameId', 'team'])

        home_df.columns = home_df.columns.map(
            lambda x: f'{x}Home' if x in columns else x)
        away_df.columns = away_df.columns.map(
            lambda x: f'{x}Away' if x in columns else x)

        try:

            df = pd.concat([home_df.set_index('gameId'),
                            away_df.set_index('gameId')], axis=1)
            df = df.drop('team', axis=1).reset_index().rename(
                columns={'gameId': 'game_id'})
            dfs.append(df)

        except ValueError:
            pass

    df = pd.concat(dfs, axis=0, ignore_index=True)
    return df


def games_transform(filepath=Path('data')):

    dfs = []
    for file in filepath.rglob('games.csv'):

        df = pd.read_csv(file, parse_dates=['start_date'])

        df.sort_values(by=['start_date'])

        df.home_line_scores = df.home_line_scores.astype(str).str.replace(
            '[', '').str.replace(']', '').str.replace('nan', '0,0,0,0')
        df.away_line_scores = df.away_line_scores.astype(str).str.replace(
            '[', '').str.replace(']', '').str.replace('nan', '0,0,0,0')

        columns = ['first_qtr_score', 'second_qtr_score',
                   'third_qtr_score', 'fourth_qtr_score']

        df[[f'home_{col}' for col in columns]] = df.home_line_scores.str.split(
            ',', expand=True).iloc[:, :4].replace('', np.nan).fillna(0).astype(int)
        df[[f'away_{col}' for col in columns]] = df.away_line_scores.str.split(
            ',', expand=True).iloc[:, :4].replace('', np.nan).fillna(0).astype(int)

        df = df.drop(['home_line_scores', 'away_line_scores',
                      'start_time_tbd'], axis=1)

        df['home_conference'] = df.groupby('home_team')['home_conference'].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))
        df['away_conference'] = df.groupby('away_team')['away_conference'].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))

        df = df.drop(['home_post_win_prob', 'away_post_win_prob'], axis=1)

        df['attendance'] = df['attendance'].fillna(0)
        df['excitement_index'] = df['excitement_index'].fillna(0)
        df['away_conference'] = df['away_conference'].fillna('MISSING')

        df = df.rename(columns={'id': 'game_id'})

        dfs.append(df)

    df[['home_points', 'away_points']] = df[[
        'home_points', 'away_points']].fillna(0)

    df = pd.concat(dfs, axis=0, ignore_index=True)
    return df


def lines_transform(filepath=Path('data')):

    dfs = []
    for file in filepath.rglob('lines.csv'):

        df = pd.read_csv(file)
        df = df.dropna(subset=['spread'])

        df['id'] = df.id.astype('int')

        df['overUnder'] = df.groupby('id')['overUnder'].transform(
            lambda x: x.fillna(x.mean()))
        df['overUnder'] = df['overUnder'].fillna(0)

        columns = ['id', 'provider', 'spread', 'overUnder']

        df = df[columns]

        df = df.pivot(index='id', columns='provider',
                      values=['spread', 'overUnder'])

        df.iloc[:, df.columns.get_level_values(0) == 'spread'] = df.iloc[:, df.columns.get_level_values(
            0) == 'spread'].T.fillna(df.iloc[:, df.columns.get_level_values(0) == 'spread'].T.mean()).T
        df.iloc[:, df.columns.get_level_values(0) == 'overUnder'] = df.iloc[:, df.columns.get_level_values(
            0) == 'overUnder'].T.fillna(df.iloc[:, df.columns.get_level_values(0) == 'overUnder'].T.mean()).T

        df.columns = df.columns.map('_'.join)

        df = df.reset_index().rename(columns={'id': 'game_id'})
        dfs.append(df)

    df = pd.concat(dfs, axis=0, ignore_index=True)

    return df


def matchup_transform(filepath=Path('data')):

    df = pd.read_csv(filepath/'matchup_aggregation.csv')

    columns = ['game_id', 'actual_spread_mean', 'actual_spread_median', 'actual_spread_std',
               'actual_over_under_mean', 'actual_over_under_median', 'actual_over_under_std']

    df = df[columns]
    df = df.fillna(0)

    return df


def pregame_transform(filepath=Path('data')):

    dfs = []
    for file in filepath.rglob('pregame.csv'):

        df = pd.read_csv(file)
        df = df[['gameId', 'spread', 'homeWinProb']]
        df['awayWinProb'] = 1 - df['homeWinProb']

        df.columns = ['game_id', 'pre_game_spread',
                      'pre_game_home_win_prob', 'pre_game_away_win_prob']

        dfs.append(df)

    df = pd.concat(dfs, axis=0, ignore_index=True)

    return df


def recruiting_position_transform(filepath=Path('data')):

    dfs = []
    for file in filepath.rglob('recruiting_position.csv'):

        df = pd.read_csv(file)

        columns = ['team', 'season', 'positionGroup',
                   'averageRating', 'totalRating', 'commits', 'averageStars']
        df = df[columns]

        df = df.groupby(['team', 'season', 'positionGroup']
                        ).sum().unstack('positionGroup').fillna(0)

        df.columns = df.columns.map('_'.join)
        df = df.reset_index()

        dfs.append(df)

    df = pd.concat(dfs, axis=0, ignore_index=True)

    return df


def recruiting_teams_transform(filepath=Path('data')):

    dfs = []
    for file in filepath.rglob('recruiting_teams.csv'):

        df = pd.read_csv(file)
        df = df.rename(columns={'year': 'season'})

        dfs.append(df)

    df = pd.concat(dfs, axis=0, ignore_index=True)

    return df


def roster_transform(filepath=Path('data')):

    dfs = []
    for file in filepath.rglob('roster_aggregation.csv'):
        df = pd.read_csv(file)
        dfs.append(df)

    df = pd.concat(dfs, axis=0, ignore_index=True)
    df = df[df.columns[~df.columns.str.contains('?', regex=False)]]

    return df


def talent_transform(filepath=Path('data')):

    dfs = []
    for file in filepath.rglob('talent.csv'):

        df = pd.read_csv(file)
        df = df.rename(columns={'year': 'season', 'school': 'team'})

        dfs.append(df)

    df = pd.concat(dfs, axis=0, ignore_index=True)
    return df


def past_weather_transform(filepath=Path('data')):

    dfs = []

    for file in (filepath/'weather').glob('*.csv'):
        df = pd.read_csv(file)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    columns = [
        'game_id', 'tempF', 'windspeedMiles', 'weatherDesc', 'precipInches', 'humidity', 'visibility',
        'pressure', 'cloudcover', 'HeatIndexF', 'DewPointF', 'WindChillF', 'WindGustMiles', 'FeelsLikeF'
    ]

    df = df[columns]
    df['weatherDesc'] = df.weatherDesc.map(
        lambda x: ast.literal_eval(x)[0]['value'])

    return df
