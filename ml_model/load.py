from extract import *
from transform import *
from datetime import datetime


def dataset(predict_week, predict_season, filepath=Path('data'), window_size=4, update_data=True, update_seasons=[pd.to_datetime('today').year]):

    if update_data:

        teams_extract(filepath)
        teams = pd.read_csv(filepath/'teams.csv')['school']

        venues_extract(filepath)
        games_extract(update_seasons, filepath)
        games_stat_extract(update_seasons, teams, filepath)
        games_advance_stats_extract(update_seasons, filepath)
        talent_extract(update_seasons, filepath)
        lines_extract(update_seasons, filepath)
        recruiting_teams_extract(update_seasons, filepath)
        recruiting_position_extract(update_seasons, filepath)
        pregame_extract(update_seasons, filepath)
        roster_aggregation(update_seasons, teams, filepath)

        # Updated once a season
        try:
            date = datetime.fromtimestamp(
                (filepath/'matchup.csv').stat().st_mtime)
            if date.year < pd.to_datetime('today').year and pd.to_datetime('today').month > 2:
                matchup_extract(filepath)
            else:
                pass
        except:
            matchup_extract(filepath)

        try:
            date = datetime.fromtimestamp(
                (filepath/'matchup_aggregation.csv').stat().st_mtime)
            if date.year < pd.to_datetime('today').year and pd.to_datetime('today').month > 2:
                matchup_aggregation(filepath)
            else:
                pass
        except:
            matchup_aggregation(filepath)

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

    games_df = games_df[(games_df.season < predict_season) | (
        (games_df.week <= predict_week) & (games_df.season == predict_season))]

    # weather_df = past_weather_transform(filepath=filepath)

    games_df['spread_target'] = games_df['away_points'] - \
        games_df['home_points']

    games_df['spread_target_rolling'] = games_df['away_points'] - \
        games_df['home_points']

    # columns used to identify game attribute
    core_columns = ['game_id', 'season', 'week', 'season_type', 'neutral_site', 'conference_game',
                    'attendance', 'venue_id', 'venue', 'home_team', 'home_conference',
                    'home_post_win_prob', 'away_team', 'away_conference',
                    'away_post_win_prob', 'start_date',
                    'spread_target']

    # Game Stats
    df = games_df.merge(game_stat_df, on='game_id', how='left')
    df = df.merge(games_advanced_stats_df, on='game_id', how='left')
    df = df.fillna(0)

    core_df = df[df.columns[df.columns.isin(core_columns)]]

    home_df = df[df.columns[(df.columns.str.contains(
        'home|Home')) & (~df.columns.isin(core_columns))]]

    away_df = df[df.columns[(df.columns.str.contains(
        'away|Away')) & (~df.columns.isin(core_columns))]]

    home_df = home_df.groupby('home_id').apply(lambda df: df.shift(
        1).rolling(window_size, min_periods=1).mean()).dropna()

    away_df = away_df.groupby('away_id').apply(lambda df: df.shift(
        1).rolling(window_size, min_periods=1).mean()).dropna()

    df = pd.concat([core_df, home_df, away_df], axis=1, join='inner')

    # Lines, Matchup, Pregame
    df = df.merge(lines_df[['game_id', 'overUnder_consensus']], on='game_id', how='left').rename(
        columns={'overUnder_consensus': 'pre_game_over_under'})

    df = df.merge(matchup_df, on='game_id', how='left').fillna(0)
    df = df.merge(pregame_df, on='game_id', how='inner')

    # Recruting
    home_recruiting_position_df = recruiting_position_df.copy()
    away_recruiting_position_df = recruiting_position_df.copy()

    home_recruiting_position_df.columns = [f'{col}Home' if col not in [
        'team', 'season'] else col for col in recruiting_position_df.columns]
    away_recruiting_position_df.columns = [f'{col}Away' if col not in [
        'team', 'season'] else col for col in recruiting_position_df.columns]

    home_recruiting_position_df = home_recruiting_position_df.rename(
        columns={'team': 'home_team'})
    away_recruiting_position_df = away_recruiting_position_df.rename(
        columns={'team': 'away_team'})

    df = df.merge(home_recruiting_position_df, on=[
                  'home_team', 'season'], how='left')
    df = df.merge(away_recruiting_position_df, on=[
                  'away_team', 'season'], how='left')

    home_recruiting_teams_df = recruiting_teams_df.copy()
    away_recruiting_teams_df = recruiting_teams_df.copy()

    home_recruiting_teams_df.columns = [f'{col}Home' if col not in [
        'team', 'season'] else col for col in recruiting_teams_df.columns]
    away_recruiting_teams_df.columns = [f'{col}Away' if col not in [
        'team', 'season'] else col for col in recruiting_teams_df.columns]

    home_recruiting_teams_df = home_recruiting_teams_df.rename(
        columns={'team': 'home_team'})
    away_recruiting_teams_df = away_recruiting_teams_df.rename(
        columns={'team': 'away_team'})

    df = df.merge(home_recruiting_teams_df, on=[
                  'home_team', 'season'], how='left')
    df = df.merge(away_recruiting_teams_df, on=[
                  'away_team', 'season'], how='left')

    # Roster
    home_roster_df = roster_df.copy()
    away_roster_df = roster_df.copy()

    home_roster_df.columns = [f'{col}Home' if col not in [
        'team', 'season'] else col for col in roster_df.columns]
    away_roster_df.columns = [f'{col}Away' if col not in [
        'team', 'season'] else col for col in roster_df.columns]

    home_roster_df = home_roster_df.rename(columns={'team': 'home_team'})
    away_roster_df = away_roster_df.rename(columns={'team': 'away_team'})

    df = df.merge(home_roster_df, on=['home_team', 'season'], how='left')
    df = df.merge(away_roster_df, on=['away_team', 'season'], how='left')

    # Talent
    home_talent_df = talent_df.copy()
    away_talent_df = talent_df.copy()

    home_talent_df.columns = [f'{col}Home' if col not in [
        'team', 'season'] else col for col in talent_df.columns]
    away_talent_df.columns = [f'{col}Away' if col not in [
        'team', 'season'] else col for col in talent_df.columns]

    home_talent_df = home_talent_df.rename(columns={'team': 'home_team'})
    away_talent_df = away_talent_df.rename(columns={'team': 'away_team'})

    df = df.merge(home_talent_df, on=['home_team', 'season'], how='left')
    df = df.merge(away_talent_df, on=['away_team', 'season'], how='left')

    # Create Rolling Mean for Spread and Result
    df['spread_diff_rolling'] = df['pre_game_spread'] - df['spread_target']
    df['spread_result_rolling'] = np.where(
        df['spread_target'] <= df['pre_game_spread'], 1, 0)

    df[['spread_diff_rolling', 'spread_result_rolling']] = df.groupby('home_id')[['spread_diff_rolling', 'spread_result_rolling']].transform(
        lambda df: df.shift(1).rolling(8, min_periods=0).mean()).fillna(df[['spread_diff_rolling', 'spread_result_rolling']].mean())

    # Weather
    # df = df.merge(weather_df, on='game_id', how='left').fillna(weather_df.mean())
    # df['weatherDesc'] = df.weatherDesc.fillna('Sunny')

    return df


model_classifier = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                 colsample_bynode=1, colsample_bytree=0.8, gamma=5, gpu_id=-1,
                                 importance_type='gain', interaction_constraints='',
                                 learning_rate=0.1, max_delta_step=0, max_depth=5,
                                 min_child_weight=10, monotone_constraints='()',
                                 n_estimators=100, n_jobs=0, num_parallel_tree=1,
                                 objective='binary:logistic', random_state=0, reg_alpha=0,
                                 reg_lambda=1, scale_pos_weight=1, subsample=1.0,
                                 tree_method='exact', validate_parameters=1, verbosity=None)
