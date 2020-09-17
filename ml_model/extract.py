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
def past_weather_extract(filepath, api_key):

    for file in filepath.rglob('games.csv'):

        # https://www.worldweatheronline.com/developer/api/docs/local-city-town-weather-api.aspx

        games_df = pd.read_csv(file, parse_dates=['start_date'])
        games_df = games_df[['start_date', 'venue_id', 'id']].rename(columns={
                                                                     'id': 'game_id'})

        venues_df = pd.read_csv(
            filepath/'venues.csv').dropna(subset=['location'])
        venues_df = venues_df[['name', 'location', 'id']].rename(
            columns={'id': 'venue_id'})

        venues_games_df = venues_df.merge(games_df, on='venue_id', how='left').dropna(
            subset=['start_date'])[['name', 'location', 'start_date', 'game_id', 'venue_id']]

        venues_games_df['location'] = venues_games_df['location'].map(
            lambda x: ast.literal_eval(x)).map(lambda x: f"{x['x']},{x['y']}")
        group = venues_games_df.groupby(['name', 'location', 'venue_id'])

        # api_key = '2a34a2fcead249289cb30207202403'
        # api_key = '0edf5aa7fc4e419681a135604202403'
        # api_key = 'fcc47f9b3f114655bd9202732202403'
        # api_key = '83470e7f0fab4852a36163453201304'
        # api_key = 'f19ba43b5a934231ad7165834201304'
        # api_key = '1123680e2ad1465f9db171642201304'

        for key, group_df in group:
            try:
                prior_df = pd.read_csv(filepath/f'weather/{str(key[2])}.csv')
                games = prior_df['game_id'].to_list()
            except:
                games = []

            print(f'reading api {key[2]}')
            dfs = []
            for date, game_id in zip(group_df['start_date'], group_df['game_id']):
                if game_id in games:
                    pass
                else:
                    url = f'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?key={api_key}&q={key[1]}&format=json&date={date.date()}&tp=24'
                    print(url)
                    data = requests.get(url).json()
                    weather_df = pd.DataFrame(
                        data['data']['weather'][0]['hourly'])
                    weather_df['name'] = key[0]
                    weather_df['start_date'] = date
                    weather_df['game_id'] = int(game_id)
                    weather_df['venue_id'] = key[2]

                    dfs.append(weather_df)

            try:
                df = pd.concat(dfs, axis=0, ignore_index=True)

                try:
                    df = pd.concat([prior_df, df], axis=0, ignore_index=True)
                    df.to_csv(filepath/f'weather/{key[2]}.csv', index=False)
                    del df, prior_df
                    print('updating')
                except:
                    df.to_csv(filepath/f'weather/{key[2]}.csv', index=False)
                    del df
                    print('new')

            except:
                del prior_df
                print('skipped')
                pass
