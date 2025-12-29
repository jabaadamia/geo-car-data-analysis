# Utility functions for data processing/cleaning

import re

def impute_by_grouping(df, impute_config, group_cols):
    # imputes missing columns using group_cols 

    for col, method in impute_config.items():

        # model statistic
        if method == 'median':
            stat_model = (
                df.dropna(subset=[col])
                  .groupby(group_cols)[col]
                  .median()
            )
        else:  # mode
            stat_model = (
                df.dropna(subset=[col])
                  .groupby(group_cols)[col]
                  .agg(lambda x: x.mode().iloc[0])
            )

        # merge model values
        df = df.merge(
            stat_model.rename(f'{col}_model_stat'),
            on=group_cols,
            how='left'
        )

        # fill from model
        df[col] = df[col].fillna(df[f'{col}_model_stat'])

        # cleanup
        df = df.drop(columns=[f'{col}_model_stat'])

    return df

# count items in list writen as text
def count_list_items(val):
    s = str(val).strip()
    if s == '':
        return 0
    parts = [p.strip() for p in re.split(r'[;,|\t\n]+', s) if p.strip() != '']
    return len(parts)