import pandas as pd

def to_naive_kyiv(s: pd.Series, tz_name: str = "Europe/Kyiv"):
    s = s.copy()

    if pd.api.types.is_integer_dtype(s.dtype) or pd.api.types.is_float_dtype(s.dtype):
        return pd.to_datetime(s, unit='ms', utc=True, errors='coerce').dt.tz_convert(tz_name).dt.tz_localize(None)
    
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.tz_convert(tz_name).dt.tz_localize(None)
    str_s = s.astype('string').str.strip()
    norm = str_s.str.replace(r'[\u200b\ufeff]', '', regex=True) \
             .str.strip() \
             .str.replace('T', ' ', regex=False) \
             .str.replace(r'Z$', '+00:00', regex=True) \
             .str.replace(r'([+-]\d{2})(\d{2})$', r'\1:\2', regex=True)
    tz_mask = norm.str.contains(r'(?:[+-]\d{2}:\d{2}|[+-]\d{4}|Z)$', regex=True, na=False)

    out = pd.Series(pd.NaT, index =s.index, dtype='datetime64[ns]')

    if tz_mask.any():
        parsed_tz = pd.to_datetime(norm[tz_mask], utc=True, errors='coerce') \
                        .dt.tz_convert(tz_name).dt.tz_localize(None)
        out.loc[tz_mask] = parsed_tz


    if (~tz_mask).any():
        parsed_naive = pd.to_datetime(norm[~tz_mask], errors='coerce')
        out.loc[~tz_mask] = parsed_naive

    return out