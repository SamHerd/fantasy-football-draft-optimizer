import pandas as pd

def load_injury_risk(path: str | None):
    if path is None:
        return pd.DataFrame(columns=['player_id','injury_risk'])
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df
