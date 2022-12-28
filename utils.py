import pandas as pd
import os


def get_data_csv(path):

    dfs = []
    for i, category in enumerate(os.listdir(path)):
        df = pd.DataFrame()
        df['filename'] = pd.Series(os.listdir(os.path.join(path, category)))
        df['path'] = pd.Series(df['filename'].apply(lambda x: os.path.join(path, category, str(x))))
        df['category'] = category
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)
