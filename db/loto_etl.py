
import re
from urllib.parse import urlparse
import pandas as pd

# ---- 入力URL ----
URLS = [
    "https://loto-life.net/csv/mini",
    "https://loto-life.net/csv/loto6",
    "https://loto-life.net/csv/loto7",
    "https://loto-life.net/csv/bingo5",
    "https://loto-life.net/csv/numbers3",
    "https://loto-life.net/csv/numbers4",
]

# numbers3/4 の短縮名
NAME_MAP = {"numbers3": "num3", "numbers4": "num4"}

def _loto_name_from_url(u: str) -> str:
    tail = urlparse(u).path.strip("/").split("/")[-1]
    return NAME_MAP.get(tail, tail)

def _read_csv_jp(u: str) -> pd.DataFrame:
    for enc in ("cp932", "sjis", "utf-8"):
        try:
            return pd.read_csv(u, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(u)

def _find_date_col(cols) -> str | None:
    for c in cols:
        if re.search(r"(開催日|抽選日)", str(c)):
            return c
    return None

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _rename_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    # ・開催日/抽選日→ds（datetime化）
    # ・第*数字 → N*
    # ・*等口数 → N*NU
    # ・*等賞金 → N*PM
    df = _normalize_columns(df)

    # 開催日/抽選日 → ds
    date_col = _find_date_col(df.columns)
    if date_col and date_col != "ds":
        df = df.rename(columns={date_col: "ds"})
    if "ds" in df.columns:
        ds_clean = (
            df["ds"].astype(str)
            .str.replace(r"\(.*?\)", "", regex=True)
            .str.replace("年", "-")
            .str.replace("月", "-")
            .str.replace("日", "")
            .str.replace("/", "-")
        )
        df["ds"] = pd.to_datetime(ds_clean, errors="coerce")

    # 列名リネーム
    ren = {}
    for c in df.columns:
        m = re.fullmatch(r"第(\d+)数字", c)
        if m:
            ren[c] = f"N{int(m.group(1))}"
            continue
        m = re.fullmatch(r"(\d+)等口数", c)
        if m:
            ren[c] = f"N{int(m.group(1))}NU"
            continue
        m = re.fullmatch(r"(\d+)等賞金", c)
        if m:
            ren[c] = f"N{int(m.group(1))}PM"
            continue
    if ren:
        df = df.rename(columns=ren)
    return df

def _melt_numbers_long(df: pd.DataFrame, loto_name: str) -> pd.DataFrame:
    # N* 列のみロング化。unique_idにN名、yに値、loto列付与。
    df = df.copy()
    df["loto"] = loto_name

    n_cols = [c for c in df.columns if re.fullmatch(r"N\d+", str(c))]
    n_cols = sorted(n_cols, key=lambda x: int(re.search(r"\d+", x).group()))
    if not n_cols:
        return pd.DataFrame(columns=["loto", "ds", "unique_id", "y"])

    id_vars = [c for c in df.columns if c not in n_cols]
    long_df = df.melt(id_vars=id_vars, value_vars=n_cols,
                      var_name="unique_id", value_name="y")
    long_df["y"] = pd.to_numeric(long_df["y"], errors="coerce")
    return long_df

def build_long_dataframe(urls: list[str]) -> pd.DataFrame:
    out_frames = []
    for u in urls:
        loto = _loto_name_from_url(u)
        raw = _read_csv_jp(u)
        norm = _rename_core_columns(raw)
        long = _melt_numbers_long(norm, loto)
        out_frames.append(long)

    # 空DFを除外してからconcat（FutureWarning回避）
    out_frames = [f for f in out_frames if f is not None and not f.empty]
    if not out_frames:
        return pd.DataFrame(columns=["loto", "num", "ds", "unique_id", "y", "CO"])
    result = pd.concat(out_frames, ignore_index=True)

    preferred = ["loto", "開催回", "ds", "unique_id", "y"]
    other = [c for c in result.columns if c not in preferred]
    cols = [c for c in preferred if c in result.columns] + other
    result = result[cols]
    return result

def _to_int64(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype("int64")

def finalize_df(df: pd.DataFrame) -> pd.DataFrame:
    # 仕様:
    #   - NaNは0埋め
    #   - 数値型はすべてint（pandasではint64）
    #   - 開催回 → num（int64）を作成し、開催回は削除
    #   - キャリーオーバー → CO（int64）を作成し、元列は削除
    #   - ボーナス数字 / ボーナス数字1 / ボーナス数字2 を削除
    df = df.copy()

    if "開催回" in df.columns:
        df["num"] = _to_int64(df["開催回"])
    if "キャリーオーバー" in df.columns:
        df["CO"] = _to_int64(df["キャリーオーバー"])
    if "y" in df.columns:
        df["y"] = _to_int64(df["y"])

    # 文字列/日時以外は一括で int64
    exclude = {"loto", "ds", "unique_id", "開催回", "キャリーオーバー", "num", "CO", "y"}
    for c in df.columns:
        if c in exclude:
            continue
        # 数値化→0埋め→int64
        df[c] = _to_int64(df[c])

    # 不要列削除
    drop_cols = [c for c in ["ボーナス数字", "ボーナス数字1", "ボーナス数字2",
                              "キャリーオーバー", "開催回"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    preferred = ["loto", "num", "ds", "unique_id", "y", "CO"]
    others = [c for c in df.columns if c not in preferred]
    df = df[[c for c in preferred if c in df.columns] + others]

    return df

def build_df_final(urls: list[str] | None = None) -> pd.DataFrame:
    """外部から呼び出すエントリポイント。"""
    _urls = urls or URLS
    df_long = build_long_dataframe(_urls)
    df_final = finalize_df(df_long)
    return df_final

if __name__ == "__main__":
    df_final = build_df_final()
    print(df_final.info())
    print(df_final.head())
