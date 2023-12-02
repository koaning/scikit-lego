def try_convert_to_standard_compliant_dataframe(df):
    if hasattr(df, '__dataframe_consortium_standard__'):
        return df.__dataframe_consortium_standard__(api_version='2023.11-beta')
    try:
        import pandas as pd
    except ModuleNotFoundError:
        pass
    else:
        if isinstance(df, pd.DataFrame):
            from dataframe_api_compat.pandas_standard import convert_to_standard_compliant_dataframe
            return convert_to_standard_compliant_dataframe(df)
    try:
        import polars as pl
    except ModuleNotFoundError:
        pass
    else:
        if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
            from dataframe_api_compat.polars_standard import convert_to_standard_compliant_dataframe
            return convert_to_standard_compliant_dataframe(df)
    return df

def try_convert_to_standard_compliant_column(df):
    if hasattr(df, '__column_consortium_standard__'):
        return df.__column_consortium_standard__(api_version='2023.11-beta')
    try:
        import pandas as pd
    except ModuleNotFoundError:
        pass
    else:
        if isinstance(df, pd.Series):
            from dataframe_api_compat.pandas_standard import convert_to_standard_compliant_column
            return convert_to_standard_compliant_column(df)
    try:
        import polars as pl
    except ModuleNotFoundError:
        pass
    else:
        if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
            from dataframe_api_compat.polars_standard import convert_to_standard_compliant_column
            return convert_to_standard_compliant_column(df)
    return df