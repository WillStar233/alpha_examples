import polars as pl


def make_forward_return(df_px: pl.DataFrame, horizon: int) -> pl.DataFrame:
    if not {"date", "symbol", "close"}.issubset(df_px.columns):
        raise ValueError("df_px must contain columns: date, symbol, close")
    out = (
        df_px.sort(["symbol", "date"]).with_columns(
            ((pl.col("close").shift(-horizon).over("symbol") / pl.col("close")) - 1.0).alias("value")
        )
        .select(["date", "symbol", "value"]).drop_nulls("value")
    )
    return out