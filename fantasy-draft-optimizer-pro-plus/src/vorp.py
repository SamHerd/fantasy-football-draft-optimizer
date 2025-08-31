import pandas as pd
from typing import Dict

def compute_replacement_levels(players: pd.DataFrame, teams: int, roster: Dict[str,int], flex_alloc: Dict[str,float]) -> Dict[str, float]:
    starters = {p: roster.get(p,0) for p in ['QB','RB','WR','TE']}
    flex = roster.get('FLEX', 0)
    eff_starters = {p: starters[p] + flex * float(flex_alloc.get(p, 0.0)) for p in starters}
    repl_rank = {p: int(round(teams * eff_starters[p])) for p in starters}
    baselines = {}
    for p in starters:
        pool = players[players['position'] == p].sort_values('proj_points', ascending=False).reset_index(drop=True)
        if len(pool) == 0:
            baselines[p] = 0.0
            continue
        idx = max(0, min(len(pool)-1, repl_rank[p]-1))  # convert 1-index rank to 0-index
        baselines[p] = float(pool.loc[idx, 'proj_points'])
    return baselines

def add_vorp(players: pd.DataFrame, baselines: Dict[str, float]) -> pd.DataFrame:
    players = players.copy()
    players['replacement_points'] = players['position'].map(baselines).fillna(0.0)
    players['VORP'] = players['proj_points'] - players['replacement_points']
    return players
