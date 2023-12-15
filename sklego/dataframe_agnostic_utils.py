"""
We use the DataFrame API Standard to write dataframe-agnostic code.

The full spec can be found here: https://data-apis.org/dataframe-api/draft/API_specification/index.html.

pandas and Polars expose the entrypoint `__dataframe_consortium_standard__` on their DataFrame and Series objects,
but only as of versions 2.1 and 0.18.13 respectively.

In order to support earlier versions, we import `convert_to_standard_compliant_dataframe` from
the `dataframe-api-compat` package.
"""

def try_convert_to_standard_compliant_dataframe(df, *, strict = False):
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
    if strict:
        raise TypeError(f"Could not convert {type(df)} to a standard compliant dataframe")
    return df

def try_convert_to_standard_compliant_column(df, *, strict = False):
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
    if strict:
        raise TypeError(f"Could not convert {type(df)} to a standard compliant dataframe")
    return df
