"""WTA v2 replay with the production recipe (v21e: points update, rank fill, surface rating from first match)."""
import paths as P
import json, sys
import confounder_harness as H
D = P.ROOT + "/wta_data"
src = sys.argv[1]                     # wta_tml.csv or wta_tml_itf.csv
H.DATA_PATHS = [f"{D}/{src}"]; H.birth = json.load(open(f"{D}/wta_birth_ordinal_v2.json"))
H.SURF_FILL = "rank"; H.MIN_SURF_MATCHES = 1
cf = json.load(open(P.OUT + "/confounder_fit_v21.json"))
H.main(out_name=f"wta2_{src.replace('.csv', '')}.npz", verbose=False, init_prior=cf["init_prior"], gscale=json.load(open(P.OUT + '/wta_fit.json'))['gscale'], pt_lam=json.load(open(P.OUT + '/wta_fit.json'))['pt_lam'], pt_c=0.7, pt_m=2.132,
       init_rank=tuple(cf["init_rank"]), age_drift=tuple(cf["age_drift"]))
print("done")
