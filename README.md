# Fantasy Draft Optimizer — Dynamic VORP + Tiers + Make-It-Back

**What’s inside**
- Dynamic **VORP** (recomputes replacement level as players are drafted)
- **Roster-need weighting** (boost positions you still need)
- Optional **playoff weeks emphasis** (wk15–wk17) if weekly projections exist
- **Positional tiers** via k-means (auto or fixed k)
- **Make-it-back probabilities** using ADP mean/std (normal model)
- Bulk/multiselect drafted input and downloadable Big Board CSV
- PPR points helper if your projections don’t include `proj_points`

**Run**
```bash
pip install -r requirements.txt
streamlit run app.py
```
