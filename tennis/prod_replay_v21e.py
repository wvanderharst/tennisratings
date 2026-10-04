"""Production v21e trajectory replay -> cf_FULL_v21e_reference.npz (rank fill, surface rating from the first match)."""
import paths as P
import json
import confounder_harness as H
cf = json.load(open(P.OUT + "/confounder_fit_v21.json"))
H.SURF_FILL = "rank"; H.MIN_SURF_MATCHES = 1
H.main(out_name="cf_FULL_v21e_reference.npz", verbose=False, init_prior=cf["init_prior"], gscale=0.8, pt_lam=1.0, pt_c=0.7, pt_m=2.132,
       init_rank=tuple(cf["init_rank"]), age_drift=tuple(cf["age_drift"]))
print("done")
