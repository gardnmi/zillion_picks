import ast
import json
import re
import time
import urllib
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.exceptions import HTTPError

from utils import flatten

################ EXTRACT ################

def teams_extract(filepath):
    
    df = pd.read_json('https://api.collegefootballdata.com/teams/fbs')
    df.to_csv(filepath/'teams.csv', index=False)


def games_extract(seasons, filepath):
    
    for season in seasons:
        df = pd.read_json(f'https://api.collegefootballdata.com/games?year={season}&seasonType=both')
        df.to_csv(filepath/f'seasons/{season}/games.csv', index=False)


def games_stat_extract(seasons, teams, filepath):
    
    for season in seasons:
        
        team_dfs = []
        for team in teams:
        
            #regular season & post season    
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
                            
                            df = pd.DataFrame(stats['stats']).set_index('category').T.add_suffix(f"{stats['homeAway'].capitalize()}")
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
            df.to_csv(filepath/f'seasons/{season}/games_stats.csv', index=False)
        except ValueError:
            pass


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

            df.to_csv(filepath/f'seasons/{season}/games_advanced_stats.csv', index=False)
        except ValueError:
            pass

def talent_extract(seasons, filepath):
    
    for season in seasons:
        
        df = pd.read_json(f'https://api.collegefootballdata.com/talent?year={season}')

        if len(df) == 0:
            pass
        else:
            df.to_csv(filepath/f'seasons/{season}/talent.csv', index=False)


def venues_extract(filepath):
    
    df = pd.read_json('https://api.collegefootballdata.com/venues')
    df.to_csv(filepath/'venues.csv', index=False)


def lines_extract(seasons, filepath):
    
    for season in seasons:
        
        url = f'https://api.collegefootballdata.com/lines?year={season}&seasonType=both'
        data = requests.get(url).json()
        
        game_dfs = []
        for game in data:
            features = pd.DataFrame({key: game[key] for key in ['id', 'homeTeam', 'awayTeam']}, index=[0])
            lines = pd.DataFrame(game['lines'])
            
            df = features.join(lines, how='outer').ffill()

            game_dfs.append(df)
        
        df = pd.concat(game_dfs, axis=0, ignore_index=True)
        df['season'] = season
        
        df.to_csv(filepath/f'seasons/{season}/lines.csv', index=False)


def recruiting_teams_extract(seasons, filepath):
    
    for season in seasons:
        
        df = pd.read_json(f'https://api.collegefootballdata.com/recruiting/teams?year={season}')
        df.to_csv(filepath/f'seasons/{season}/recruiting_teams.csv', index=False)


def recruiting_position_extract(seasons, filepath):
    
    for season in seasons:
        
        df = pd.read_json(f'https://api.collegefootballdata.com/recruiting/groups?startYear={season}&endYear={season}')
        df['season'] = season

        df.to_csv(filepath/f'seasons/{season}/recruiting_position.csv', index=False)


def pregame_extract(seasons, filepath):
    
    for season in seasons:
    
        df1 = pd.read_json(f'https://api.collegefootballdata.com/metrics/wp/pregame?year={season}&seasonType=regular')
        df2 = pd.read_json(f'https://api.collegefootballdata.com/metrics/wp/pregame?year={season}&seasonType=postseason')
    
        df = pd.concat([df1, df2], axis=0, ignore_index=True)
        df.to_csv(filepath/f'seasons/{season}/pregame.csv', index=False)


def matchup_extract(filepath):

    dfs = []
    for file in filepath.rglob('games.csv'):
        df = pd.read_csv(file)    
        dfs.append(df)

    games_df = pd.concat(dfs, axis=0, ignore_index=True)
    games_df[['home_team', 'away_team']] = np.sort(games_df[['home_team', 'away_team']])

    dfs = []
    for row in games_df[['home_team', 'away_team']].drop_duplicates().itertuples(index=False):
        url = f'https://api.collegefootballdata.com/teams/matchup?team1={urllib.parse.quote(row.home_team)}&team2={urllib.parse.quote(row.away_team)}&minYear=2005'
        data = requests.get(url).json()
        
        df = pd.DataFrame(data['games'])

        dfs.append(df)

    df = pd.concat(dfs, axis=0, ignore_index=True)
    df.to_csv(filepath/'matchup.csv', index=False)


def matchup_aggregation(filepath):

    dfs = []
    for file in filepath.rglob('games.csv'):
        df = pd.read_csv(file)    
        dfs.append(df)
   
    games_df = pd.concat(dfs, axis=0, ignore_index=True)
    matchup_df = pd.read_csv(filepath/'matchup.csv')

    dfs = []

    for row in games_df[['id', 'season', 'home_team', 'away_team']].itertuples(index=False):

        df = matchup_df[(matchup_df['homeTeam'] == row.home_team) & (matchup_df['awayTeam'] == row.away_team)]
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


def roster_aggregation(seasons, teams, filepath):
    
    for season in seasons:
        
        team_dfs = []
        for team in teams:

            df = pd.read_json(f'https://api.collegefootballdata.com/roster?team={urllib.parse.quote(team)}&year={season}')

            if len(df) == 0:
                pass
            
            else:
                df[['weight', 'height', 'year']] = df[['weight', 'height', 'year']].fillna(df[['weight', 'height', 'year']].mean())
                df = df.groupby('position')['weight', 'height', 'year'].mean()
                

                df = pd.concat([
                    df.loc[:,'weight'].add_prefix(f"{df.loc[:,'weight'].name} "),
                    df.loc[:,'height'].add_prefix(f"{df.loc[:,'height'].name} "),
                    df.loc[:,'year'].add_prefix(f"{df.loc[:,'year'].name} ")
                    ]).to_frame().T

                df['team'] = team
                
                team_dfs.append(df)
        try:   
            df = pd.concat(team_dfs, axis=0, ignore_index=True)
            df['season'] = season
            df.to_csv(filepath/f'seasons/{season}/roster_aggregation.csv', index=False)
        except ValueError:
            pass


def past_weather_extract(filepath, api_key):
    
    for file in filepath.rglob('games.csv'):

        # https://www.worldweatheronline.com/developer/api/docs/local-city-town-weather-api.aspx

        games_df = pd.read_csv(file, parse_dates=['start_date'])
        games_df = games_df[['start_date', 'venue_id', 'id']].rename(columns={'id':'game_id'})

        venues_df = pd.read_csv(filepath/'venues.csv').dropna(subset=['location'])
        venues_df = venues_df[['name', 'location', 'id']].rename(columns={'id':'venue_id'})

        venues_games_df = venues_df.merge(games_df, on='venue_id', how='left').dropna(subset=['start_date'])[['name', 'location', 'start_date', 'game_id', 'venue_id']]

        venues_games_df['location'] = venues_games_df['location'].map(lambda x: ast.literal_eval(x)).map(lambda x: f"{x['x']},{x['y']}")
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
                    weather_df = pd.DataFrame(data['data']['weather'][0]['hourly'])
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


################ TRANSFORM ################

def game_stat_transform(filepath):
    
    dfs = []
    for file in filepath.rglob('games_stats.csv'):


        df = pd.read_csv(file)

        split_columns = ['thirdDownEffAway', 'thirdDownEffHome',
                        'fourthDownEffAway', 'fourthDownEffHome',
                        'completionAttemptsAway', 'completionAttemptsHome']

        for column in split_columns:
            
            success_column = f'{column[:-4]}Success{column[-4:]}'
            attempt_column = f'{column[:-4]}Attempts{column[-4:]}'

            df[[success_column, attempt_column]] = df[column].str.split('-', expand=True).iloc[:, :2].astype(float)
            df[column] = df[success_column] /  df[attempt_column]

        split_columns = ['totalPenaltiesYardsAway', 'totalPenaltiesYardsHome']

        for column in split_columns:
            
            new_column = re.sub('Yards', '', column)
            
            df[[new_column, column]] = df[column].str.split('-', n=1, expand=True).astype(float)

        time_columns = ['possessionTimeAway', 'possessionTimeHome']

        for column in time_columns:
            
            df[column] = pd.to_timedelta(df[column] +':00').dt.total_seconds()

        df = df.fillna(0)

        df = df.drop(['home', 'away'], axis=1)

        dfs.append(df)

    df = pd.concat(dfs, axis=0, ignore_index=True)
    return df


def games_advanced_stats_transform(filepath):
    
    dfs = []

    for files in zip(filepath.rglob('games_advanced_stats.csv'), filepath.rglob('games.csv')):

        df = pd.read_csv(files[0])
        games_df = pd.read_csv(files[1])

        df = df.drop(['week', 'opponent', 'season'], axis=1)

        home_df = df.merge(games_df[['id', 'home_team']], left_on=['gameId', 'team'], right_on=['id', 'home_team'], how='inner').drop(['id', 'home_team'], axis=1)
        away_df = df.merge(games_df[['id', 'away_team']], left_on=['gameId', 'team'], right_on=['id', 'away_team'], how='inner').drop(['id', 'away_team'], axis=1)

        columns = home_df.columns.difference(['gameId', 'team'])

        home_df.columns = home_df.columns.map(lambda x: f'{x}Home' if x in columns else x)
        away_df.columns = away_df.columns.map(lambda x: f'{x}Away' if x in columns else x)
        
        try:

            df = pd.concat([home_df.set_index('gameId'), away_df.set_index('gameId')], axis=1)
            df = df.drop('team', axis=1).reset_index().rename(columns={'gameId':'game_id'})
            dfs.append(df)

        except ValueError:
            pass

    df = pd.concat(dfs, axis=0, ignore_index=True)
    return df


def games_transform(filepath):
    
    dfs = []
    for file in filepath.rglob('games.csv'):

        df = pd.read_csv(file, parse_dates=['start_date'])

        df.sort_values(by=['start_date'])

        df.home_line_scores = df.home_line_scores.astype(str).str.replace('[', '').str.replace(']', '').str.replace('nan', '0,0,0,0')
        df.away_line_scores = df.away_line_scores.astype(str).str.replace('[', '').str.replace(']', '').str.replace('nan', '0,0,0,0')

        columns = ['first_qtr_score', 'second_qtr_score', 'third_qtr_score', 'fourth_qtr_score']

        df[[f'home_{col}' for col in columns]] = df.home_line_scores.str.split(',', expand=True).iloc[:,:4].replace('', np.nan).fillna(0).astype(int)
        df[[f'away_{col}' for col in columns]] = df.away_line_scores.str.split(',', expand=True).iloc[:,:4].replace('', np.nan).fillna(0).astype(int)

        df = df.drop(['home_line_scores', 'away_line_scores', 'start_time_tbd'], axis=1)

        df['home_conference'] = df.groupby('home_team')['home_conference'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))
        df['away_conference'] = df.groupby('away_team')['away_conference'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))

        df = df.drop(['home_post_win_prob','away_post_win_prob'], axis=1)

        df['attendance'] = df['attendance'].fillna(0)
        df['excitement_index'] = df['excitement_index'].fillna(0)
        df['away_conference'] = df['away_conference'].fillna('MISSING')

        df = df.rename(columns={'id':'game_id'})

        dfs.append(df)
    
    df[['home_points', 'away_points']] = df[['home_points', 'away_points']].fillna(0)

    df = pd.concat(dfs, axis=0, ignore_index=True)
    return df


def lines_transform(filepath):
    
    dfs = []
    for file in filepath.rglob('lines.csv'):

        df = pd.read_csv(file)
        df = df.dropna(subset=['spread'])

        df['id'] = df.id.astype('int')

        df['overUnder'] = df.groupby('id')['overUnder'].transform(lambda x: x.fillna(x.mean()))
        df['overUnder'] = df['overUnder'].fillna(0)

        columns = ['id', 'provider', 'spread', 'overUnder']

        df = df[columns] 

        df = df.pivot(index='id', columns='provider', values=['spread', 'overUnder'])
        
        df.iloc[:, df.columns.get_level_values(0)=='spread'] = df.iloc[:, df.columns.get_level_values(0)=='spread'].T.fillna(df.iloc[:, df.columns.get_level_values(0)=='spread'].T.mean()).T
        df.iloc[:, df.columns.get_level_values(0)=='overUnder'] = df.iloc[:, df.columns.get_level_values(0)=='overUnder'].T.fillna(df.iloc[:, df.columns.get_level_values(0)=='overUnder'].T.mean()).T
        
        df.columns = df.columns.map('_'.join)

        df = df.reset_index().rename(columns={'id':'game_id'})
        dfs.append(df)

    df = pd.concat(dfs, axis=0, ignore_index=True)

    return df


def matchup_transform(filepath):
    
    df = pd.read_csv(filepath/'matchup_aggregation.csv')

    columns = ['game_id', 'actual_spread_mean', 'actual_spread_median', 'actual_spread_std', 
            'actual_over_under_mean', 'actual_over_under_median', 'actual_over_under_std']

    df = df[columns]
    df = df.fillna(0)

    return df


def pregame_transform(filepath):
    
    dfs = []
    for file in filepath.rglob('pregame.csv'):

        df = pd.read_csv(file)
        df = df[['gameId', 'spread', 'homeWinProb']]
        df['awayWinProb'] = 1 - df['homeWinProb']
        
        df.columns = ['game_id', 'pre_game_spread', 'pre_game_home_win_prob', 'pre_game_away_win_prob']

        dfs.append(df)
    
    df = pd.concat(dfs, axis=0, ignore_index=True)

    return df


def recruiting_position_transform(filepath):
    
    dfs = []
    for file in filepath.rglob('recruiting_position.csv'):

        df = pd.read_csv(file)

        columns = ['team', 'season', 'positionGroup', 'averageRating', 'totalRating', 'commits', 'averageStars']
        df = df[columns]

        df = df.groupby(['team', 'season', 'positionGroup']).sum().unstack('positionGroup').fillna(0)

        df.columns = df.columns.map('_'.join)
        df = df.reset_index()

        dfs.append(df)

    df = pd.concat(dfs, axis=0, ignore_index=True)

    return df


def recruiting_teams_transform(filepath):
    
    dfs = []
    for file in filepath.rglob('recruiting_teams.csv'):

        df = pd.read_csv(file)
        df = df.rename(columns={'year':'season'})

        dfs.append(df)
    
    df = pd.concat(dfs, axis=0, ignore_index=True)

    return df


def roster_transform(filepath):
    
    dfs = []
    for file in filepath.rglob('roster_aggregation.csv'):
        df = pd.read_csv(file)
        dfs.append(df)

    df = pd.concat(dfs, axis=0, ignore_index=True)
    df = df[df.columns[~df.columns.str.contains('?', regex=False)]]

    return df


def talent_transform(filepath):
    
    dfs = []
    for file in filepath.rglob('talent.csv'):

        df = pd.read_csv(file)
        df = df.rename(columns={'year':'season', 'school':'team'})

        dfs.append(df)
    
    df = pd.concat(dfs, axis=0, ignore_index=True)
    return df


def past_weather_transform(filepath):
    
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
    df['weatherDesc'] = df.weatherDesc.map(lambda x: ast.literal_eval(x)[0]['value'])

    return df

################ LOAD ################

def update_data(filepath = Path('data'), seasons = [pd.to_datetime('today').year]):

        teams_extract(filepath)        
        teams=pd.read_csv(filepath/'teams.csv')['school']
        
        venues_extract(filepath)
        games_extract(seasons, filepath)
        games_stat_extract(seasons, teams, filepath)
        games_advance_stats_extract(seasons, filepath)
        talent_extract(seasons, filepath)    
        lines_extract(seasons, filepath)
        recruiting_teams_extract(seasons, filepath)
        recruiting_position_extract(seasons, filepath)
        pregame_extract(seasons, filepath)
        
        # Update This once a Season
        # matchup_extract(filepath)
        # matchup_aggregation(filepath)
        
        roster_aggregation(seasons, teams, filepath)


def dataset(predict_week, predict_season, filepath = Path('data'), window_size=4):

    games_df = games_transform(filepath=filepath)
    game_stat_df = game_stat_transform(filepath=filepath)
    games_advanced_stats_df = games_advanced_stats_transform(filepath=filepath)
    lines_df = lines_transform(filepath=filepath)
    matchup_df = matchup_transform(filepath=filepath)
    pregame_df = pregame_transform(filepath=filepath)
    recruiting_position_df = recruiting_position_transform(filepath=filepath)
    recruiting_teams_df = recruiting_teams_transform(filepath=filepath)
    roster_df = roster_transform(filepath=filepath)
    talent_df = talent_transform(filepath=filepath)

    games_df = games_df[(games_df.season < predict_season) | ((games_df.week <= predict_week) & (games_df.season == predict_season))]
    # weather_df = past_weather_transform(filepath=filepath)

    games_df['spread_target'] = games_df['away_points'] - games_df['home_points']       
    games_df['over_under_target'] = games_df['away_points'] + games_df['home_points'] 

        # 1 Picking Home Team Wins and 0 Picking Away Team wins
        # df = games_df[['game_id', 'home_points', 'away_points']].merge(pregame_df[['game_id', 'pre_game_spread']])
        # games_df['classifier_target'] = (((df['home_points'] - df['away_points']) * -1) < df['pre_game_spread']).astype(int)

    # columns used to identify game attribute
    core_columns = ['game_id', 'season', 'week', 'season_type', 'neutral_site', 'conference_game',
                    'attendance', 'venue_id', 'venue', 'home_team', 'home_conference',
                    'home_post_win_prob', 'away_team', 'away_conference',
                    'away_post_win_prob', 'excitement_index', 'start_date', 
                    'spread_target', 'over_under_target']

    # Game Stats
    df = games_df.merge(game_stat_df, on='game_id', how='left')
    df = df.merge(games_advanced_stats_df, on='game_id', how='left')
    df = df.fillna(0)

    core_df = df[df.columns[df.columns.isin(core_columns)]]
    home_df = df[df.columns[(df.columns.str.contains('home|Home')) & (~df.columns.isin(core_columns))]]
    away_df = df[df.columns[(df.columns.str.contains('away|Away')) & (~df.columns.isin(core_columns))]]

    home_df = home_df.groupby('home_id').apply(lambda df: df.shift(1).rolling(window_size, min_periods=1).mean()).dropna()
    away_df = away_df.groupby('away_id').apply(lambda df: df.shift(1).rolling(window_size, min_periods=1).mean()).dropna()

    df = pd.concat([core_df, home_df, away_df], axis=1, join='inner')

    # Lines, Matchup, Pregame
    df = df.merge(lines_df[['game_id', 'overUnder_consensus']], on='game_id', how='left').rename(columns={'overUnder_consensus':'pre_game_over_under'})
    df = df.merge(matchup_df, on='game_id', how='left').fillna(0)
    df = df.merge(pregame_df, on='game_id', how='inner')

    # Recruting
    home_recruiting_position_df = recruiting_position_df.copy()
    away_recruiting_position_df = recruiting_position_df.copy()

    home_recruiting_position_df.columns = [f'{col}Home' if col not in ['team', 'season'] else col for col in recruiting_position_df.columns]
    away_recruiting_position_df.columns = [f'{col}Away' if col not in ['team', 'season'] else col for col in recruiting_position_df.columns]

    home_recruiting_position_df = home_recruiting_position_df.rename(columns={'team':'home_team'})
    away_recruiting_position_df = away_recruiting_position_df.rename(columns={'team':'away_team'})

    df = df.merge(home_recruiting_position_df, on=['home_team', 'season'], how='left')#.fillna(home_recruiting_position_df.mean())
    df = df.merge(away_recruiting_position_df, on=['away_team', 'season'], how='left')#.fillna(away_recruiting_position_df.mean())

    home_recruiting_teams_df = recruiting_teams_df.copy()
    away_recruiting_teams_df = recruiting_teams_df.copy()

    home_recruiting_teams_df.columns = [f'{col}Home' if col not in ['team', 'season'] else col for col in recruiting_teams_df.columns]
    away_recruiting_teams_df.columns = [f'{col}Away' if col not in ['team', 'season'] else col for col in recruiting_teams_df.columns]

    home_recruiting_teams_df = home_recruiting_teams_df.rename(columns={'team':'home_team'})
    away_recruiting_teams_df = away_recruiting_teams_df.rename(columns={'team':'away_team'})

    df = df.merge(home_recruiting_teams_df, on=['home_team', 'season'], how='left')#.fillna(home_recruiting_teams_df.mean())
    df = df.merge(away_recruiting_teams_df, on=['away_team', 'season'], how='left')#.fillna(away_recruiting_teams_df.mean())

    # Roster
    home_roster_df = roster_df.copy()
    away_roster_df = roster_df.copy()

    home_roster_df.columns = [f'{col}Home' if col not in ['team', 'season'] else col for col in roster_df.columns]
    away_roster_df.columns = [f'{col}Away' if col not in ['team', 'season'] else col for col in roster_df.columns]

    home_roster_df = home_roster_df.rename(columns={'team':'home_team'})
    away_roster_df = away_roster_df.rename(columns={'team':'away_team'})

    df = df.merge(home_roster_df, on=['home_team', 'season'], how='left')#.fillna(home_roster_df.mean())
    df = df.merge(away_roster_df, on=['away_team', 'season'], how='left')#.fillna(away_roster_df.mean())

    # Talent
    home_talent_df = talent_df.copy()
    away_talent_df = talent_df.copy()

    home_talent_df.columns = [f'{col}Home' if col not in ['team', 'season'] else col for col in talent_df.columns]
    away_talent_df.columns = [f'{col}Away' if col not in ['team', 'season'] else col for col in talent_df.columns]

    home_talent_df = home_talent_df.rename(columns={'team':'home_team'})
    away_talent_df = away_talent_df.rename(columns={'team':'away_team'})

    df = df.merge(home_talent_df, on=['home_team', 'season'], how='left')#.fillna(home_talent_df.mean())
    df = df.merge(away_talent_df, on=['away_team', 'season'], how='left')#.fillna(away_talent_df.mean())

    # Weather
    # df = df.merge(weather_df, on='game_id', how='left').fillna(weather_df.mean())
    # df['weatherDesc'] = df.weatherDesc.fillna('Sunny')

    # Clean Up
    df[df.columns[df.columns.str.contains('home|Home')]] = df[df.columns[df.columns.str.contains('home|Home')]].groupby('home_team', as_index=False, group_keys=False).apply(lambda x: x.fillna(x.mean()))
    df[df.columns[df.columns.str.contains('away|Away')]] = df[df.columns[df.columns.str.contains('away|Away')]].groupby('away_team', as_index=False, group_keys=False).apply(lambda x: x.fillna(x.mean()))
    df = df.dropna(axis=1, thresh=int(len(df) * .9))
    df = df.fillna(0)
    df[df.select_dtypes('bool').columns] = df.select_dtypes('bool').astype('int')

    # return df.drop(['regression_target', 'classifier_target'], axis=1), df[['regression_target', 'classifier_target']]

    return df.drop(['spread_target', 'over_under_target'], axis=1), df[['spread_target', 'over_under_target']]
