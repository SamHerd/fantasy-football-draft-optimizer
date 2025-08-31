
import streamlit as st
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from math import erf, sqrt, ceil

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.vorp import compute_replacement_levels, add_vorp

st.set_page_config(page_title="Fantasy Draft Optimizer — Pro+", layout="wide")

st.title("🏈 Fantasy Draft Optimizer — Pro+")
st.caption("Dynamic VORP • Roster Needs • Playoff Boost • Tiers • Make-It-Back • Tier Cliffs • Bench Upside • Strategy Presets • Bye/Depth • Explain • Scenario Sim")

# -----------------------------
# Sidebar: League & Weights
# -----------------------------
cfg_path = Path("config.yaml")
cfg = yaml.safe_load(open(cfg_path)) if cfg_path.exists() else {}

teams = st.sidebar.number_input("Teams in League", min_value=6, max_value=20, value=int(cfg.get('league',{}).get('teams',12)))
roster_defaults = cfg.get('league',{}).get('roster', {'QB':1,'RB':2,'WR':2,'TE':1,'FLEX':1})

st.sidebar.subheader("Starters per Position")
qb = st.sidebar.number_input("QB", 0, 3, int(roster_defaults.get('QB',1)))
rb = st.sidebar.number_input("RB", 0, 5, int(roster_defaults.get('RB',2)))
wr = st.sidebar.number_input("WR", 0, 5, int(roster_defaults.get('WR',2)))
te = st.sidebar.number_input("TE", 0, 3, int(roster_defaults.get('TE',1)))
flex = st.sidebar.number_input("FLEX (RB/WR/TE)", 0, 3, int(roster_defaults.get('FLEX',1)))

flex_alloc_defaults = cfg.get('flex_allocation', {'RB':0.5,'WR':0.4,'TE':0.1})
st.sidebar.subheader("FLEX Allocation")
fa_rb = st.sidebar.slider("FLEX share to RB", 0.0, 1.0, float(flex_alloc_defaults.get('RB',0.5)), key="fa_rb")
fa_wr = st.sidebar.slider("FLEX share to WR", 0.0, 1.0, float(flex_alloc_defaults.get('WR',0.4)), key="fa_wr")
fa_te = st.sidebar.slider("FLEX share to TE", 0.0, 1.0, float(flex_alloc_defaults.get('TE',0.1)), key="fa_te")

st.sidebar.subheader("Weights")
w_defaults = cfg.get('weights', {'projected_points':0.45,'vorp':0.35,'risk':0.1,'adp_value':0.1})
w_proj = st.sidebar.slider("Projected Points", 0.0, 1.0, float(w_defaults.get('projected_points',0.45)), key="w_proj")
w_vorp = st.sidebar.slider("VORP", 0.0, 1.0, float(w_defaults.get('vorp',0.35)), key="w_vorp")
w_risk = st.sidebar.slider("Risk (lower is better)", 0.0, 1.0, float(w_defaults.get('risk',0.1)), key="w_risk")
w_adp  = st.sidebar.slider("ADP Value (proximity to pick)", 0.0, 1.0, float(w_defaults.get('adp_value',0.1)), key="w_adp")

st.sidebar.subheader("Roster Need Weighting")
need_weight = st.sidebar.slider("Need weight (boost positions you still need)", 0.0, 1.0, 0.25, key="need_weight")

st.sidebar.subheader("Playoff Weeks Emphasis (optional)")
playoff_weight = st.sidebar.slider("Playoff boost weight", 0.0, 1.0, 0.0, key="playoff_weight")

st.sidebar.subheader("Bench Upside Mode")
enable_upside = st.sidebar.checkbox("Enable Bench Upside Mode", value=False, key="enable_upside")
upside_weight = st.sidebar.slider("Upside weight (ceiling - floor)", 0.0, 1.0, 0.35, key="upside_weight")
upside_start_round = st.sidebar.number_input("Upside starts at round #", min_value=1, max_value=30, value=9, key="upside_start_round")
upside_ramp_rounds = st.sidebar.number_input("Ramp length (rounds)", min_value=1, max_value=10, value=4, key="upside_ramp_rounds")

st.sidebar.subheader("Make-It-Back Settings")
next_pick = st.sidebar.number_input("Your NEXT pick (overall #)", min_value=1, max_value=400, value=int(cfg.get('draft',{}).get('pick_number',12)), key="next_pick")
picks_until_following = st.sidebar.number_input("Picks until your FOLLOWING pick", min_value=1, max_value=400, value=int(max(1, 2*(teams-1))), key="picks_until_following")

st.sidebar.subheader("Tiering")
tier_k = st.sidebar.number_input("Tiers per position (0 = auto)", min_value=0, max_value=10, value=4, key="tier_k")
cliff_threshold = st.sidebar.number_input("Cliff alert threshold (# players left in best tier)", min_value=1, max_value=10, value=1, key="cliff_threshold")

# -----------------------------
# Strategy Presets
# -----------------------------
st.sidebar.subheader("Strategy Presets")
presets = {
    "Balanced": {"w_proj":0.45,"w_vorp":0.35,"w_risk":0.10,"w_adp":0.10,"need_weight":0.25,"playoff_weight":0.0,"enable_upside":False},
    "Anchor RB": {"w_proj":0.48,"w_vorp":0.40,"w_risk":0.08,"w_adp":0.04,"need_weight":0.30,"playoff_weight":0.0,"enable_upside":False},
    "Zero RB": {"w_proj":0.44,"w_vorp":0.32,"w_risk":0.08,"w_adp":0.16,"need_weight":0.35,"playoff_weight":0.0,"enable_upside":True},
    "Safe": {"w_proj":0.42,"w_vorp":0.34,"w_risk":0.18,"w_adp":0.06,"need_weight":0.20,"playoff_weight":0.0,"enable_upside":False},
    "Upside": {"w_proj":0.38,"w_vorp":0.34,"w_risk":0.06,"w_adp":0.12,"need_weight":0.20,"playoff_weight":0.0,"enable_upside":True},
    "Playoff Push": {"w_proj":0.38,"w_vorp":0.34,"w_risk":0.08,"w_adp":0.10,"need_weight":0.20,"playoff_weight":0.30,"enable_upside":True},
}
preset_name = st.sidebar.selectbox("Choose a preset", list(presets.keys()), index=0)
if st.sidebar.button("Apply Preset"):
    p = presets[preset_name]
    for k, v in p.items():
        st.session_state[k] = v

st.sidebar.markdown("---")
st.sidebar.caption("Upload your latest CSVs or use the samples below.")

# -----------------------------
# Data Uploads
# -----------------------------
proj_file = st.file_uploader("Projections CSV (PPR points preferred)", type=['csv'], key='proj')
adp_file  = st.file_uploader("ADP CSV", type=['csv'], key='adp')
risk_file = st.file_uploader("Injury Risk CSV (optional)", type=['csv'], key='risk')

def read_csv_or_sample(upload, sample_path):
    if upload is not None:
        return pd.read_csv(upload)
    return pd.read_csv(sample_path)

proj_df = read_csv_or_sample(proj_file, "data/sample_projections.csv")
adp_df  = read_csv_or_sample(adp_file, "data/sample_adp.csv")
risk_df = read_csv_or_sample(risk_file, "data/sample_injury_risk.csv")

# -----------------------------
# OPTIONAL: Compute proj_points from raw stats (PPR)
# -----------------------------
if 'proj_points' not in proj_df.columns:
    stat_cols = set(c.lower() for c in proj_df.columns)
    needed = {'rec','rec_yds','rec_td','rush_yds','rush_td','pass_yds','pass_td'}
    if needed.issubset(stat_cols):
        def col(c):
            for cc in proj_df.columns:
                if cc.lower() == c: return cc
            return c
        PPR = 1.0
        proj_df['proj_points'] = (
            PPR * proj_df[col('rec')].fillna(0)
            + 0.1 * proj_df[col('rec_yds')].fillna(0)
            + 6.0 * proj_df[col('rec_td')].fillna(0)
            + 0.1 * proj_df[col('rush_yds')].fillna(0)
            + 6.0 * proj_df[col('rush_td')].fillna(0)
            + 0.04 * proj_df[col('pass_yds')].fillna(0)
            + 4.0 * proj_df[col('pass_td')].fillna(0)
            - 2.0 * (proj_df[col('ints')].fillna(0) if 'ints' in stat_cols else 0)
        )
        st.info("Computed PPR proj_points from raw stat columns.")

# -----------------------------
# Session state
# -----------------------------
if "drafted" not in st.session_state:
    st.session_state.drafted = []
if "my_picks" not in st.session_state:
    st.session_state.my_picks = []
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

# -----------------------------
# Helpers
# -----------------------------
def zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    std = s.std(ddof=0)
    if std == 0 or s.isna().all():
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / (std)

def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2)))

def prob_available_at_pick(adp: float, adp_std: float, pick: float) -> float:
    sd = adp_std if (pd.notna(adp_std) and adp_std > 0) else 8.0
    if pd.isna(adp):
        return 0.0
    z = (pick - adp) / sd
    drafted_before = normal_cdf(z)
    return float(max(0.0, min(1.0, 1.0 - drafted_before)))

def current_round_from_pick(pick_overall: int, teams: int) -> int:
    return int(ceil(pick_overall / float(teams)))

def upside_ramp_factor(curr_round: int, start_round: int, ramp_len: int) -> float:
    if curr_round < start_round:
        return 0.0
    step = curr_round - start_round + 1
    return float(max(0.0, min(1.0, step / float(max(1, ramp_len)))))

# -----------------------------
# Live Draft Controls
# -----------------------------
st.header("🎯 Live Draft Controls")
with st.expander("Manage Drafted, My Roster, and Watchlist", expanded=True):
    colA, colB, colC = st.columns(3)

    with colA:
        st.subheader("Drafted (remove from pool)")
        available_names = sorted(set(proj_df['player_name']) - set(st.session_state.drafted))
        draft_multi = st.multiselect("Add drafted", options=available_names, key="draft_multi")
        bulk = st.text_area("Paste (one per line)", height=100, key="draft_bulk")
        if st.button("➕ Add to Drafted"):
            pasted = [s.strip() for s in st.session_state['draft_bulk'].splitlines() if s.strip()]
            st.session_state.drafted = sorted(set(st.session_state.drafted).union(st.session_state['draft_multi']).union(pasted))
        if st.button("♻️ Reset Drafted"):
            st.session_state.drafted = []

    with colB:
        st.subheader("🧍 My Roster")
        my_options = sorted(set(proj_df['player_name']) - set(st.session_state.my_picks))
        mine_multi = st.multiselect("Add to My Roster", options=my_options, key="my_multi")
        if st.button("➕ Add Picks"):
            st.session_state.my_picks = sorted(set(st.session_state.my_picks).union(st.session_state.get('my_multi', [])))
        if st.button("♻️ Reset My Roster"):
            st.session_state.my_picks = []

    with colC:
        st.subheader("⭐ Watchlist")
        wl_options = sorted(set(proj_df['player_name']) - set(st.session_state.watchlist))
        wl_multi = st.multiselect("Add to Watchlist", options=wl_options, key="wl_multi")
        if st.button("➕ Add Stars"):
            st.session_state.watchlist = sorted(set(st.session_state.watchlist).union(st.session_state.get('wl_multi', [])))
        if st.button("🗑️ Clear Watchlist"):
            st.session_state.watchlist = []

# -----------------------------
# Core builders
# -----------------------------
def build_board(proj_df_in: pd.DataFrame, adp_df_in: pd.DataFrame, risk_df_in: pd.DataFrame,
                drafted_names: list, my_picks_names: list,
                teams: int, roster: dict, flex_alloc: dict,
                w_proj: float, w_vorp: float, w_risk: float, w_adp: float,
                need_weight: float, playoff_weight: float,
                enable_upside: bool, upside_weight: float, upside_start_round: int, upside_ramp_rounds: int,
                next_pick: int, picks_until_following: int, tier_k: int):
    # Remaining pool
    remaining_proj = proj_df_in[~proj_df_in['player_name'].isin(drafted_names)].copy()
    remaining_adp  = adp_df_in[~adp_df_in['player_name'].isin(drafted_names)].copy()

    # Baselines / VORP
    baselines_live = compute_replacement_levels(remaining_proj, teams, roster, flex_alloc)
    players_live = add_vorp(remaining_proj, baselines_live)

    # Merge
    df = players_live.merge(remaining_adp[['player_id','adp','adp_std']], on='player_id', how='left')
    if 'injury_risk' in risk_df_in.columns:
        df = df.merge(risk_df_in[['player_id','injury_risk']], on='player_id', how='left')
    else:
        df['injury_risk'] = np.nan
    df['injury_risk'] = df['injury_risk'].fillna(0.12)

    # ADP proximity
    df['adp_gap'] = df['adp'] - int(next_pick)
    df['adp_score'] = -np.abs(df['adp_gap'])

    # Round & ramp
    curr_round = current_round_from_pick(int(next_pick), int(teams))
    ramp = upside_ramp_factor(curr_round, int(upside_start_round), int(upside_ramp_rounds)) if enable_upside else 0.0

    # z features
    df['z_proj'] = zscore(df['proj_points'])
    df['z_vorp'] = zscore(df['VORP'])
    df['z_risk'] = zscore(-df['injury_risk'])
    df['z_adp']  = zscore(df['adp_score'])

    # Upside feature
    if {'proj_ceiling','proj_floor'}.issubset(df.columns):
        df['upside_span'] = (df['proj_ceiling'] - df['proj_floor']).clip(lower=0)
        df['z_upside'] = zscore(df['upside_span'])
    else:
        df['z_upside'] = 0.0

    # Effective risk weight
    risk_weight_eff = w_risk * (1.0 - 0.5 * ramp)

    # Base contributions
    df['contrib_proj']  = w_proj * df['z_proj']
    df['contrib_vorp']  = w_vorp * df['z_vorp']
    df['contrib_risk']  = risk_weight_eff * df['z_risk']
    df['contrib_adp']   = w_adp  * df['z_adp']
    df['contrib_upside']= ramp * upside_weight * df['z_upside']

    df['DraftScore'] = df['contrib_proj'] + df['contrib_vorp'] + df['contrib_risk'] + df['contrib_adp'] + df['contrib_upside']

    # Positional need contributions
    my_roster_df = proj_df_in[proj_df_in['player_name'].isin(my_picks_names)]
    my_counts = my_roster_df['position'].value_counts().to_dict()
    need_targets = {
        'QB': float(roster.get('QB',0)),
        'RB': float(roster.get('RB',0)) + (float(roster.get('FLEX',0)) > 0) * float(flex_alloc.get('RB',0)),
        'WR': float(roster.get('WR',0)) + (float(roster.get('FLEX',0)) > 0) * float(flex_alloc.get('WR',0)),
        'TE': float(roster.get('TE',0)) + (float(roster.get('FLEX',0)) > 0) * float(flex_alloc.get('TE',0)),
    }
    def need_score_for_pos(pos: str) -> float:
        have = float(my_counts.get(pos, 0))
        need = float(need_targets.get(pos, 0))
        gap = need - have
        if gap > 0:   return 1.0
        if gap == 0:  return 0.3
        return -0.2
    df['need_pos_score'] = df['position'].map(lambda p: need_score_for_pos(p)).fillna(0.0)
    df['contrib_need'] = need_weight * df['need_pos_score']
    df['DraftScore'] += df['contrib_need']

    # Playoff emphasis
    weekly_cols = [c for c in proj_df_in.columns if c.lower().startswith('wk')]
    playoff_cols = [c for c in weekly_cols if c.lower() in ('wk15','wk16','wk17')]
    if playoff_weight > 0 and weekly_cols and playoff_cols:
        wk_map = proj_df_in[['player_id'] + playoff_cols].set_index('player_id')
        df = df.join(wk_map, on='player_id', how='left')
        df['playoff_sum'] = df[playoff_cols].sum(axis=1, numeric_only=True)
        df['z_playoff'] = zscore(df['playoff_sum'].fillna(0.0))
        df['contrib_playoff'] = playoff_weight * df['z_playoff']
        df['DraftScore'] += df['contrib_playoff']
    else:
        df['contrib_playoff'] = 0.0

    # Probabilities
    df['Prob_Avail_Next'] = df.apply(lambda r: prob_available_at_pick(r['adp'], r.get('adp_std', np.nan), float(next_pick)), axis=1)
    df['Prob_Avail_Following'] = df.apply(lambda r: prob_available_at_pick(r['adp'], r.get('adp_std', np.nan), float(next_pick) + float(picks_until_following)), axis=1)

    # Tiers
    def compute_tiers(df_in: pd.DataFrame, k: int):
        use = df_in[['proj_points','VORP','injury_risk','adp']].copy()
        use['injury_risk'] = -use['injury_risk']
        use = use.fillna(use.mean(numeric_only=True))
        for c in use.columns:
            s = use[c].astype(float)
            sd = s.std(ddof=0)
            use[c] = (s - s.mean()) / (sd if sd != 0 else 1.0)
        kk = k
        if kk == 0:
            if len(use) < 3:
                kk = 1
            else:
                best_k, best_score = 2, -1.0
                for test_k in range(2, min(6, len(use)) + 1):
                    km = KMeans(n_clusters=test_k, n_init=10, random_state=42).fit(use)
                    try:
                        sc = silhouette_score(use, km.labels_)
                    except Exception:
                        sc = -1.0
                    if sc > best_score:
                        best_k, best_score = test_k, sc
                kk = best_k
        if kk <= 1 or len(use) < kk:
            labels = np.zeros(len(use), dtype=int)
        else:
            km = KMeans(n_clusters=kk, n_init=10, random_state=42).fit(use)
            labels = km.labels_
        out = df_in.copy()
        out['cluster'] = labels
        clust_scores = out.groupby('cluster')[['proj_points','VORP']].mean().sum(axis=1)
        order = clust_scores.sort_values(ascending=False).index.tolist()
        clust_to_tier = {cl: i+1 for i, cl in enumerate(order)}
        out['Tier'] = out['cluster'].map(clust_to_tier)
        return out.drop(columns=['cluster'])

    tiered_list = []
    for pos in ['RB','WR','TE','QB']:
        pos_df = df[df['position'] == pos].copy()
        if pos_df.empty: continue
        tiered_list.append(compute_tiers(pos_df, tier_k))
    if tiered_list:
        tiered_all = pd.concat(tiered_list, ignore_index=True)
        df = df.merge(tiered_all[['player_id','Tier']], on='player_id', how='left')
    else:
        df['Tier'] = np.nan

    df['⭐'] = df['player_name'].isin(st.session_state.watchlist)

    # Final board
    cols = ['player_id','player_name','team','position','bye_week','proj_points','VORP','injury_risk','adp','Tier','DraftScore','Prob_Avail_Next','Prob_Avail_Following','⭐',
            'contrib_proj','contrib_vorp','contrib_risk','contrib_adp','contrib_upside','contrib_need','contrib_playoff']
    board = df.sort_values('DraftScore', ascending=False).reset_index(drop=True)[cols]

    return board, baselines_live, curr_round, ramp

# -----------------------------
# Build current board
# -----------------------------
roster = {'QB':int(qb),'RB':int(rb),'WR':int(wr),'TE':int(te),'FLEX':int(flex)}
flex_alloc = {'RB':st.session_state['fa_rb'],'WR':st.session_state['fa_wr'],'TE':st.session_state['fa_te']}

board, baselines_live, curr_round, ramp = build_board(
    proj_df, adp_df, risk_df,
    st.session_state.drafted, st.session_state.my_picks,
    teams, roster, flex_alloc,
    st.session_state['w_proj'], st.session_state['w_vorp'], st.session_state['w_risk'], st.session_state['w_adp'],
    st.session_state['need_weight'], st.session_state['playoff_weight'],
    st.session_state['enable_upside'], st.session_state['upside_weight'], st.session_state['upside_start_round'], st.session_state['upside_ramp_rounds'],
    st.session_state['next_pick'], st.session_state['picks_until_following'], st.session_state['tier_k']
)

st.markdown(f"**Replacement baselines (live, proj points):** `{baselines_live}`")

# -----------------------------
# Cliff Alerts
# -----------------------------
st.header("⛰️ Tier Cliff Alerts")
alerts = []
for pos in ['RB','WR','TE','QB']:
    pos_df = board[board['position'] == pos]
    if pos_df.empty or pos_df['Tier'].isna().all():
        continue
    best_tier = int(pos_df['Tier'].min())
    remaining_in_best = pos_df[pos_df['Tier'] == best_tier]
    if len(remaining_in_best) <= int(cliff_threshold):
        names = ", ".join(remaining_in_best['player_name'].head(5).tolist())
        alerts.append(f"{pos} Tier {best_tier}: only {len(remaining_in_best)} left ({names})")
if alerts:
    for a in alerts:
        st.warning(a)
else:
    st.info("No immediate tier cliffs based on your threshold.")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Big Board", "📊 Tiers", "📅 Roster & Byes", "🧠 Explain", "🧪 Scenario Sim"])

# Styling helper for Big Board
def highlight_tier_cliffs(df_show: pd.DataFrame):
    styles = pd.DataFrame('', index=df_show.index, columns=df_show.columns)
    for pos in ['RB','WR','TE','QB']:
        pos_mask = df_show['position'] == pos
        pos_slice = df_show[pos_mask]
        if pos_slice.empty or pos_slice['Tier'].isna().all():
            continue
        for tier_val, group in pos_slice.groupby('Tier'):
            idxs = group.index.tolist()
            if len(idxs) > 0:
                last_idx = idxs[-1]
                styles.loc[last_idx, :] = 'background-color: rgba(255,165,0,0.25);'
    if '⭐' in df_show.columns:
        watch_idxs = df_show.index[df_show['⭐'] == True]
        for i in watch_idxs:
            styles.loc[i, 'player_name'] = 'font-weight: bold;'
    return styles

with tab1:
    st.subheader(f"Best Available (Dynamic) — Round {curr_round}  |  Upside ramp: {ramp:.2f}")
    styled = board.style.apply(highlight_tier_cliffs, axis=None).format({
        'proj_points':'{:.1f}','VORP':'{:.1f}','injury_risk':'{:.2f}','adp':'{:.1f}',
        'DraftScore':'{:.3f}','Prob_Avail_Next':'{:.2%}','Prob_Avail_Following':'{:.2%}'})
    st.dataframe(styled, use_container_width=True)
    st.download_button("Download Current Big Board (CSV)", data=board.to_csv(index=False), file_name="big_board.csv", mime="text/csv")
    st.caption("Rows shaded orange = last player of a tier within a position. Bold names are on your ⭐ Watchlist.")

with tab2:
    st.subheader("Positional Tiers")
    cols_layout = st.columns(2)
    for i, pos in enumerate(['RB','WR','TE','QB']):
        pos_df = board[board['position'] == pos].copy()
        if pos_df.empty:
            continue
        with cols_layout[i % 2]:
            st.markdown(f"**{pos} Tiers**")
            st.dataframe(pos_df[['player_name','team','proj_points','VORP','injury_risk','adp','Tier']].style.format({'proj_points':'{:.1f}','VORP':'{:.1f}','injury_risk':'{:.2f}','adp':'{:.1f}'}), use_container_width=True)

with tab3:
    st.subheader("My Depth & Bye Weeks")
    my_df = proj_df[proj_df['player_name'].isin(st.session_state.my_picks)].copy()
    if my_df.empty:
        st.info("Add players to **My Roster** in the Live Draft Controls to see depth and bye analysis.")
    else:
        depth = my_df.groupby('position')['player_name'].count().reindex(['QB','RB','WR','TE']).fillna(0).astype(int)
        st.write("**Depth by Position (count)**")
        st.bar_chart(depth)
        st.write("**Bye Weeks (count)**")
        byes = my_df.groupby(['position','bye_week'])['player_name'].count().unstack(0).fillna(0).astype(int)
        st.dataframe(byes, use_container_width=True)
        st.caption("Watch for heavy overlap at thin positions (e.g., TE).")

with tab4:
    st.subheader("Explain a Player's Rank")
    pick = st.selectbox("Choose a player", options=board['player_name'].tolist())
    row = board[board['player_name'] == pick].iloc[0]
    expl = pd.DataFrame({
        'Component': ['Proj','VORP','Risk','ADP','Upside','Need','Playoff'],
        'Contribution': [row['contrib_proj'],row['contrib_vorp'],row['contrib_risk'],row['contrib_adp'],row['contrib_upside'],row['contrib_need'],row['contrib_playoff']]
    }).sort_values('Contribution', ascending=False)
    st.dataframe(expl.style.format({'Contribution':'{:.3f}'}), use_container_width=True)
    st.caption(f"DraftScore = sum of contributions = {row['DraftScore']:.3f}")

with tab5:
    st.subheader("Scenario Simulator — If I Take X Now, What’s Likely at My Next Pick?")
    candidates = st.multiselect("Choose 1–3 candidates to compare", options=board['player_name'].head(50).tolist(), max_selections=3)
    prob_cut = st.slider("Treat players with Prob_Avail_Next ≥ this as ‘likely available’", 0.0, 1.0, 0.50, 0.05)
    topn = st.number_input("Show top N expected options", 3, 20, 8)
    if st.button("Run Scenario Sim") and candidates:
        cols = st.columns(len(candidates))
        for i, name in enumerate(candidates):
            with cols[i]:
                st.markdown(f"**If you take _{name}_ now:**")
                # simulate: add candidate to drafted + my roster
                drafted_sim = st.session_state.drafted + [name]
                my_sim = list(set(st.session_state.my_picks + [name]))
                board_sim, _, _, _ = build_board(
                    proj_df, adp_df, risk_df,
                    drafted_sim, my_sim,
                    teams, roster, flex_alloc,
                    st.session_state['w_proj'], st.session_state['w_vorp'], st.session_state['w_risk'], st.session_state['w_adp'],
                    st.session_state['need_weight'], st.session_state['playoff_weight'],
                    st.session_state['enable_upside'], st.session_state['upside_weight'], st.session_state['upside_start_round'], st.session_state['upside_ramp_rounds'],
                    st.session_state['next_pick'], st.session_state['picks_until_following'], st.session_state['tier_k']
                )
                expected = board_sim[board_sim['Prob_Avail_Next'] >= prob_cut].copy().head(int(topn))
                st.dataframe(expected[['player_name','position','adp','Prob_Avail_Next','DraftScore','Tier']].style.format({'adp':'{:.1f}','Prob_Avail_Next':'{:.0%}','DraftScore':'{:.3f}'}), use_container_width=True)
                pos_mix = expected['position'].value_counts()
                if not pos_mix.empty:
                    st.caption("Likely available (pos mix): " + ", ".join([f"{p}:{c}" for p,c in pos_mix.items()]))
