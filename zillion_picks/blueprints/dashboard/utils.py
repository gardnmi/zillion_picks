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

    cols = ['start_date', 'home_team', 'away_team', 'pre_game_spread', 'regression_spread_pred', 'predicted_spread',
            'classification_spread_pred', 'classification_confidence', 'actual_spread', 'regression_spread_result', 'classification_spread_result'
            ]

    col_names = {
        'start_date': 'Date',
        'home_team': 'Home Team',
        'away_team': 'Away Team',
        'pre_game_spread': 'Sportsbook Spread (Avg)',
        'regression_spread_pred': 'Regression Model Pick',
        'predicted_spread': 'Regression Model Spread',
        'classification_spread_pred': 'Classification Model Pick',
        'classification_confidence': 'Classification Model Confidence',
        'actual_spread': 'Final Game Result',
        'regression_spread_result': 'Regression Model Result',
        'classification_spread_result': 'Classification Model Result', }

    if week == 'postseason':
        df['week'] = 'Post Season'
    else:
        pass

    df['start_date'] = pd.to_datetime(df['start_date'])
    df['start_date'] = df['start_date'].dt.tz_convert('US/Central')
    df['start_date'] = df['start_date'].dt.strftime('%B %d, %Y')
    df['classification_confidence'] = df.classification_confidence.map(
        '{:.1%}'.format)

    df['regression_spread_pred'] = np.where(
        df['regression_spread_pred'].eq(1), df['home_team'], df['away_team'])

    df['classification_spread_pred'] = np.where(
        df['classification_spread_pred'].eq(1), df['home_team'], df['away_team'])

    df = df[cols]
    df = df.rename(columns=col_names)

    df = df.fillna('Awaiting Game Results')

    return df


def get_results(df, filepath, season, week):
    # Current Week Results
    try:
        regression_dict = df['regression_spread_result'].value_counts(
        ).to_dict()
        classification_dict = df['classification_spread_result'].value_counts(
        ).to_dict()

        regression_spread_result = {}
        for key, value in regression_dict.items():
            regression_spread_result[key] = value / \
                sum(regression_dict.values())

        classification_spread_result = {}
        for key, value in classification_dict.items():
            classification_spread_result[key] = value / \
                sum(classification_dict.values())
    except:
        regression_spread_result = {}
        classification_spread_result = {}

    try:
        # Season To Date Results
        files_df = pd.DataFrame([(file.stem, file) for file in filepath.rglob(
            './*.csv')], columns=['Season_Week', 'file_path'])

        files_df[['Season', 'Week']] = files_df.Season_Week.str.split(
            '_', n=2, expand=True)

        files_df = files_df.replace('postseason', np.nan)

        files_df['Week Nums'] = files_df.groupby('Season').transform(
            lambda x: x.reset_index().index+1)['Week']

        if week == '01' and len(regression_spread_result) == 0:
            files_df = files_df[files_df['Season'].astype(
                int) == (int(season) - 1)]

        elif week == 'postseason' and len(regression_spread_result) > 0:
            files_df = files_df[files_df['Season'].astype(
                int) == df['season'].unique()[0]]

        elif week == 'postseason' and len(regression_spread_result) == 0:
            files_df = files_df[(files_df['Week Nums'].isnull()) & (
                files_df['Season'].astype(int) == df['season'].unique()[0])]

        elif len(regression_spread_result) > 0:
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

        std_regression_spread_dict = all_df['regression_spread_result'].value_counts(
        ).to_dict()
        std_classification_spread_dict = all_df['classification_spread_result'].value_counts(
        ).to_dict()

        std_regression_spread_results = {}
        for key, value in std_regression_spread_dict.items():
            std_regression_spread_results[key] = value / \
                sum(std_regression_spread_dict.values())

        std_classification_spread_results = {}
        for key, value in std_classification_spread_dict.items():
            std_classification_spread_results[key] = value / \
                sum(std_classification_spread_dict.values())
    except:
        std_regression_spread_results = {}
        std_classification_spread_results = {}

    return regression_spread_result, classification_spread_result, std_regression_spread_results, std_classification_spread_results
