import pandas as pd


def xlsx_to_csv(in_file_name, out_file_name):
    """Extract data from xls, transform, load to csv"""
    dfs = _extract(in_file_name)
    transformed_df = _transform(dfs)
    _load(transformed_df, out_file_name)


def _extract(file_name):
    xls = pd.ExcelFile(file_name)
    value_names = ['uo', 'vo', 'so', 'thetao']
    dfs = {value_name: pd.read_excel(xls, value_name) for value_name in value_names}
    return dfs


def _transform(dfs):
    dfs = {value_name: _transform_df(df, value_name) for value_name, df in dfs.items()}

    on_cols = ['time', 'depth', 'lat', 'lon']
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on=on_cols)
    return merged_df


def _transform_df(df, value_name):
    lat = list(range(1, 92))
    df.columns = ['time', 'depth', 'lat'] + lat
    return df.iloc[1:, :].melt(
        id_vars=['time', 'depth', 'lat'],
        value_vars=lat,
        var_name='lon',
        value_name=value_name,
    )


def _load(df, file_name):
    df.to_csv(file_name, index=False)
