import ast
import json
import urllib
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from requests.exceptions import HTTPError
from helper import flatten
import logging
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, after_log

# Setup Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
file_handler = logging.FileHandler('extract.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10), after=after_log(logger, logging.DEBUG))
def teams_extract(filepath):

    df = pd.read_json('https://api.collegefootballdata.com/teams/fbs')
    df.to_csv(filepath/'teams.csv', index=False)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10), after=after_log(logger, logging.DEBUG))
def games_extract(seasons, filepath):

    for season in seasons:
        df = pd.read_json(
            f'https://api.collegefootballdata.com/games?year={season}&seasonType=both')

        # Update postseason week to be max week plus 1 for modeling reasons
        df['week'] = np.where(df['season_type'] ==
                              'postseason', df['week'].max() + 1, df['week'])
        df = df.sort_values(by='start_date')

        df.to_csv(filepath/f'seasons/{season}/games.csv', index=False)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10), after=after_log(logger, logging.DEBUG))
def games_stat_extract(seasons, teams, filepath):

    for season in seasons:

        team_dfs = []
        for team in teams:

            # regular season & post season
            urls = [
                f'https://api.collegefootballdata.com/games/teams?year={season}&seasonType=regular&team={urllib.parse.quote(team)}',
                f'https://api.collegefootballdata.com/games/teams?year={season}&seasonType=postseason&team={urllib.parse.quote(team)}'
            ]

            for url in urls:

                data = requests.get(url).json()

                if len(data) > 0:

                    game_dfs = []
                    for game in data:

                        game_id = game['id']
                        stat_dfs = []
                        for stats in game['teams']:

                            df = pd.DataFrame(stats['stats']).set_index(
                                'category').T.add_suffix(f"{stats['homeAway'].capitalize()}")
                            df[stats['homeAway']] = stats['school']

                            stat_dfs.append(df)

                        df = pd.concat(stat_dfs, axis=1)
                        df['game_id'] = game_id

                        # Fix Duplicate Column Issue
                        df = df.loc[:, ~df.columns.duplicated()]

                        game_dfs.append(df)

                    df = pd.concat(game_dfs, axis=0, ignore_index=True)
                    team_dfs.append(df)

        try:
            df = pd.concat(team_dfs, axis=0, ignore_index=True)
            # Drop Duplicate Games (Same Game for Each Team)
            df = df.drop_duplicates()
            df.to_csv(
                filepath/f'seasons/{season}/games_stats.csv', index=False)
        except ValueError:
            pass


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10), after=after_log(logger, logging.DEBUG))
def games_advance_stats_extract(seasons, filepath):

    for season in seasons:
        url = f'https://api.collegefootballdata.com/stats/game/advanced?year={season}&seasonType=both'
        data = requests.get(url).json()

        dfs = []
        for game in data:
            d = flatten(game)
            df = pd.DataFrame(d, index=[0])
            dfs.append(df)

        try:
            df = pd.concat(dfs, axis=0, ignore_index=True)
            df['season'] = season

            df.to_csv(
                filepath/f'seasons/{season}/games_advanced_stats.csv', index=False)
        except ValueError:
            pass


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10), after=after_log(logger, logging.DEBUG))
def talent_extract(seasons, filepath):

    for season in seasons:

        df = pd.read_json(
            f'https://api.collegefootballdata.com/talent?year={season}')

        if len(df) == 0:
            pass
        else:
            df.to_csv(filepath/f'seasons/{season}/talent.csv', index=False)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10), after=after_log(logger, logging.DEBUG))
def venues_extract(filepath):

    df = pd.read_json('https://api.collegefootballdata.com/venues')
    df.to_csv(filepath/'venues.csv', index=False)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10), after=after_log(logger, logging.DEBUG))
def lines_extract(seasons, filepath):

    for season in seasons:

        url = f'https://api.collegefootballdata.com/lines?year={season}&seasonType=both'
        data = requests.get(url).json()

        game_dfs = []
        for game in data:
            features = pd.DataFrame(
                {key: game[key] for key in ['id', 'homeTeam', 'awayTeam']}, index=[0])
            lines = pd.DataFrame(game['lines'])

            df = features.join(lines, how='outer').ffill()

            game_dfs.append(df)

        df = pd.concat(game_dfs, axis=0, ignore_index=True)
        df['season'] = season

        df.to_csv(filepath/f'seasons/{season}/lines.csv', index=False)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10), after=after_log(logger, logging.DEBUG))
def recruiting_teams_extract(seasons, filepath):

    for season in seasons:

        df = pd.read_json(
            f'https://api.collegefootballdata.com/recruiting/teams?year={season}')
        df.to_csv(
            filepath/f'seasons/{season}/recruiting_teams.csv', index=False)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10), after=after_log(logger, logging.DEBUG))
def recruiting_position_extract(seasons, filepath):

    for season in seasons:

        df = pd.read_json(
            f'https://api.collegefootballdata.com/recruiting/groups?startYear={season}&endYear={season}')
        df['season'] = season

        df.to_csv(
            filepath/f'seasons/{season}/recruiting_position.csv', index=False)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10), after=after_log(logger, logging.DEBUG))
def pregame_extract(seasons, filepath):

    for season in seasons:

        df1 = pd.read_json(
            f'https://api.collegefootballdata.com/metrics/wp/pregame?year={season}&seasonType=regular')
        df2 = pd.read_json(
            f'https://api.collegefootballdata.com/metrics/wp/pregame?year={season}&seasonType=postseason')

        df = pd.concat([df1, df2], axis=0, ignore_index=True)
        df.to_csv(filepath/f'seasons/{season}/pregame.csv', index=False)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10), after=after_log(logger, logging.DEBUG))
def matchup_extract(filepath):

    dfs = []
    for file in filepath.rglob('games.csv'):
        df = pd.read_csv(file)
        dfs.append(df)

    games_df = pd.concat(dfs, axis=0, ignore_index=True)
    games_df[['home_team', 'away_team']] = np.sort(
        games_df[['home_team', 'away_team']])

    dfs = []
    for row in games_df[['home_team', 'away_team']].drop_duplicates().itertuples(index=False):
        url = f'https://api.collegefootballdata.com/teams/matchup?team1={urllib.parse.quote(row.home_team)}&team2={urllib.parse.quote(row.away_team)}&minYear=2005'
        data = requests.get(url).json()

        df = pd.DataFrame(data['games'])

        dfs.append(df)

    df = pd.concat(dfs, axis=0, ignore_index=True)
    df.to_csv(filepath/'matchup.csv', index=False)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10), after=after_log(logger, logging.DEBUG))
def matchup_aggregation(filepath):

    dfs = []
    for file in filepath.rglob('games.csv'):
        df = pd.read_csv(file)
        dfs.append(df)

    games_df = pd.concat(dfs, axis=0, ignore_index=True)
    matchup_df = pd.read_csv(filepath/'matchup.csv')

    dfs = []

    for row in games_df[['id', 'season', 'home_team', 'away_team']].itertuples(index=False):

        df = matchup_df[(matchup_df['homeTeam'] == row.home_team)
                        & (matchup_df['awayTeam'] == row.away_team)]
        df = df[df['season'] < row.season]

        if len(df) == 0:
            pass

        else:
            df['actual_spread'] = df['homeScore'] - df['awayScore']
            df['actual_over_under'] = df['homeScore'] + df['awayScore']

            df = df.groupby(['homeTeam', 'awayTeam'])[['actual_spread', 'actual_over_under']].agg(
                actual_spread_mean=('actual_spread', 'mean'),
                actual_spread_median=('actual_spread', 'median'),
                actual_spread_std=('actual_spread', 'std'),
                actual_over_under_mean=('actual_over_under', 'mean'),
                actual_over_under_median=('actual_over_under', 'median'),
                actual_over_under_std=('actual_over_under', 'std')
            ).reset_index()

            df['game_id'] = row.id
            df['season'] = row.season

            dfs.append(df)

    df = pd.concat(dfs, axis=0, ignore_index=True)
    df.to_csv(filepath/'matchup_aggregation.csv', index=False)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10), after=after_log(logger, logging.DEBUG))
def roster_aggregation(seasons, teams, filepath):

    for season in seasons:

        team_dfs = []
        for team in teams:

            df = pd.read_json(
                f'https://api.collegefootballdata.com/roster?team={urllib.parse.quote(team)}&year={season}')

            if len(df) == 0:
                pass

            else:
                df[['weight', 'height', 'year']] = df[['weight', 'height', 'year']].fillna(
                    df[['weight', 'height', 'year']].mean())
                df = df.groupby('position')['weight', 'height', 'year'].mean()

                df = pd.concat([
                    df.loc[:, 'weight'].add_prefix(
                        f"{df.loc[:,'weight'].name} "),
                    df.loc[:, 'height'].add_prefix(
                        f"{df.loc[:,'height'].name} "),
                    df.loc[:, 'year'].add_prefix(f"{df.loc[:,'year'].name} ")
                ]).to_frame().T

                df['team'] = team

                team_dfs.append(df)
        try:
            df = pd.concat(team_dfs, axis=0, ignore_index=True)
            df['season'] = season
            df.to_csv(
                filepath/f'seasons/{season}/roster_aggregation.csv', index=False)
        except ValueError:
            pass


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10), after=after_log(logger, logging.DEBUG))
def roster(seasons, teams, filepath):

    for season in seasons:

        team_dfs = []
        for team in teams:

            df = pd.read_json(
                f'https://api.collegefootballdata.com/roster?team={urllib.parse.quote(team)}&year={season}')

            if len(df) == 0:
                pass

            else:
                df['team'] = team
                team_dfs.append(df)
        try:
            df = pd.concat(team_dfs, axis=0, ignore_index=True)
            df['season'] = season
            df.to_csv(filepath/f'seasons/{season}/roster.csv', index=False)
        except ValueError:
            pass


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10), after=after_log(logger, logging.DEBUG))
def games_stat_extract_alt(seasons=[folder.name for folder in Path('data/seasons/').iterdir() if folder.is_dir()], filepath=Path('data/')):

    for season in seasons:
        games = pd.read_csv(filepath/f'seasons/{season}/games.csv').id

        games_df = []
        for game_id in games:

            url = f'https://api.collegefootballdata.com/games/teams?year={season}&gameId={game_id}'
            data = requests.get(url).json()
            print(url)
            if len(data) > 0:

                subdata_dfs = []
                for subdata in data:
                    teams_dfs = []
                    for team in subdata['teams']:

                        df = pd.DataFrame(
                            team['stats']).set_index('category').T
                        df['game_id'] = game_id
                        df['school'] = team['school']
                        df['home_away'] = team['homeAway']
                        df['points_scored'] = team['points']

                        teams_dfs.append(df)

                    df = pd.concat(teams_dfs, axis=0)
                    df = df.fillna(0)

                    subdata_dfs.append(df)

                df = pd.concat(subdata_dfs, axis=0, ignore_index=True)
                games_df.append(df)

            else:
                pass

        df = pd.concat(games_df, axis=0, ignore_index=True)

        # Transform Columns
        split_columns = ['thirdDownEff', 'fourthDownEff', 'completionAttempts']
        for column in split_columns:

            success_column = f'{column}Success'
            attempt_column = f'{column}Attempts'

            df[[success_column, attempt_column]] = df[column].str.split(
                '-', expand=True).iloc[:, :2].astype(float)
            df[column] = df[success_column] / df[attempt_column]

        df[['totalPenalties', 'totalPenaltiesYards']] = df['totalPenaltiesYards'].str.split(
            '-', n=1, expand=True).astype(float)

        df['possessionTime'] = df['possessionTime'].fillna('00:00')
        df['possessionTime'] = df['possessionTime'].replace(0, '00:00')
        df['possessionTime'] = pd.to_timedelta(
            df['possessionTime'] + ':00').dt.total_seconds()

        df = df.fillna(0)
        df.to_csv(
            filepath/f'seasons/{season}/games_stats_alt.csv', index=False)
