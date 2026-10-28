from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home() / "Desktop"                   # OASIS3 / ADNI / NACC
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
YEAR = 365.25


def to_days(s):
    """date →  from 1930-01-01 """
    return (pd.to_datetime(s, errors="coerce") - pd.Timestamp("1930-01-01")).dt.days


def clean_sob(s):
    """Keep sobCDR in 0–18; everything else (-1, -4, 88 ... missing codes) → NaN"""
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s.between(0, 18))

def make_pid(name, s):
    # Unify pid format: cohort prefix + strip trailing '.0' from float IDs
    return name + "_" + pd.Series(s).astype(str).str.replace(r"\.0$", "", regex=True).values

# Step 1: sobCDR
# OASIS-3
def sob_oasis():
    S = ROOT / "OASIS3/OAS3_data_files/OASIS3_data_files/scans"
    f = next(S.glob("UDSb4*/resources/csv/files/OASIS3_UDSb4_cdr.csv"))
    c = pd.read_csv(f)
    return pd.DataFrame(dict(pid=c.OASISID,
                             t=c.days_to_visit,
                             sobcdr=clean_sob(c.CDRSUM)))


# ADNI
def sob_adni():
    c = pd.read_parquet(ROOT / "ADNI/ADNIMERGE2/ADNI_parquet/CDR.parquet",
                        columns=["RID", "VISDATE", "CDRSB", "ORIGPROT"])
    c = c[c.ORIGPROT != "TEAM"]
    return pd.DataFrame(dict(pid=c.RID,
                             t=to_days(c.VISDATE),
                             sobcdr=clean_sob(c.CDRSB),
                             phase=c.ORIGPROT))


# NACC
def sob_nacc():
    u = pd.read_parquet(ROOT / "NACC/investigator_nacc75.parquet",
                        columns=["NACCID", "VISITDATE", "CDRSUM"])
    return pd.DataFrame(dict(pid=u.NACCID,
                             t=to_days(u.VISITDATE),
                             sobcdr=clean_sob(u.CDRSUM)))


# merge
parts = []
for name, fn in [("OASIS", sob_oasis), ("ADNI", sob_adni), ("NACC", sob_nacc)]:
    d = fn().assign(cohort=name)
    d["pid"] = make_pid(name, d.pid)
    parts.append(d)

sob = (pd.concat(parts, ignore_index=True)
         .dropna(subset=["t", "sobcdr"])
         .drop_duplicates(["pid", "t"]))

# print(sob.groupby("cohort").agg(rows=("pid", "size"), people=("pid", "nunique")))
# print(sob.groupby("cohort").sobcdr.describe().round(2))
# print(sob.dtypes)

# Step 2: MRI information
def mri_oasis():
    S = ROOT / "OASIS3/OAS3_data_files/OASIS3_data_files/scans"
    f = next(S.glob("FS-Freesurfer*/resources/csv/files/OASIS3_Freesurfer_output.csv"))
    fs = pd.read_csv(f)
    fs = fs[fs["FS QC Status"].astype(str).str.lower().str.startswith("pass")]
    t = fs.MR_session.str.extract(r"_d(\d+)$")[0].astype(float)
    return pd.DataFrame(dict(pid=fs.Subject, t=t))


def mri_adni():
    fs = pd.read_parquet(ROOT / "ADNI/ADNIMERGE2/ADNI_parquet/ADSP_PHC_T1_FS.parquet",
                         columns=["RID", "PHC_SCANDATE"])
    return pd.DataFrame(dict(pid=fs.RID, t=to_days(fs.PHC_SCANDATE)))


def mri_nacc():
    fs = pd.read_csv(ROOT / "NACC/adsp_phc/extracted/Imaging_T1/NACC_ADSP_PHC_T1_FS_2026.02.18.csv",
                     usecols=["NACCID", "Scan_Date"])
    return pd.DataFrame(dict(pid=fs.NACCID, t=to_days(fs.Scan_Date)))


parts = []
for name, fn in [("OASIS", mri_oasis), ("ADNI", mri_adni), ("NACC", mri_nacc)]:
    d = fn().assign(cohort=name)
    d["pid"] = make_pid(name, d.pid)
    parts.append(d)

mri = (pd.concat(parts, ignore_index=True)
         .dropna(subset=["t"])
         .drop_duplicates(["pid", "t"]))

# baseline MRI = first for everyone
mri_bl = mri.sort_values("t").drop_duplicates("pid", keep="first")

# print(mri.groupby("cohort").agg(scans=("pid", "size"), people=("pid", "nunique")))
# print("baseline MRI:", mri_bl.groupby("cohort").size().to_dict())
# print("MRI no CDR:",
      # (~mri_bl.pid.isin(sob.pid)).groupby(mri_bl.cohort).sum().to_dict())

# Step 3 + 4
def nearest(left, right, tol, direction="nearest"):
    """For each (pid, t) in left, find the closest row in right for the same pid within tol days
       (direction: "nearest" = either side, "backward" = on/before t, "forward" = on/after t)"""
    l = left[["pid", "t"]].drop_duplicates().astype({"t": float}).sort_values("t")
    r = right.rename(columns={"t": "t_r"}).astype({"t_r": float}).sort_values("t_r")
    return pd.merge_asof(l, r, left_on="t", right_on="t_r", by="pid",
                         direction=direction, tolerance=float(tol))


def build(adni_old_only, fu_from="cdr"):
    """baseline = each participant's first MRI
       fu_from: "cdr" = follow-up window measured from the baseline sobCDR date;
                "mri" = measured from the MRI date"""
    # ---- Step 3: baseline sobCDR (within ±90 days of MRI) ----
    left = mri_bl[["pid", "cohort", "t"]].sort_values("t")
    right = (sob[["pid", "t", "sobcdr"]]
             .rename(columns={"t": "t_bl", "sobcdr": "sob_bl"})
             .sort_values("t_bl"))
    bl = pd.merge_asof(left, right, left_on="t", right_on="t_bl",
                       by="pid", direction="nearest", tolerance=90.0)
    bl = bl.dropna(subset=["sob_bl"])

    # ---- Step 4: follow-up sobCDR(After baseline sobCDR 1.75–2.25 years)----
    later = sob[["pid", "t", "sobcdr"]].rename(columns={"t": "t_fu", "sobcdr": "sob_fu"})
    fu = bl[["pid", "t", "t_bl"]].merge(later, on="pid")
    start = fu.t if fu_from == "mri" else fu.t_bl
    fu["fu_years"] = (fu.t_fu - start) / YEAR
    fu = fu[fu.fu_years.between(1.75, 2.25)].copy()
    fu["dist"] = (fu.fu_years - 2).abs()
    fu = fu.sort_values("dist").drop_duplicates(["pid", "t"])

    d = bl.merge(fu[["pid", "t", "t_fu", "sob_fu", "fu_years"]], on=["pid", "t"])

    # Step 5: age(MRI datetime)
    a = nearest(d, age_tbl, 3 * YEAR)
    a["age"] = a.age + (a.t - a.t_r) / YEAR
    d = d.merge(a[["pid", "t", "age"]], on=["pid", "t"])

    # ---- Step 5: BMI (closest within 1 year before baseline; if none, fall back to within 30 days after the scan) ----
    # Use pre-scan values only, so every baseline feature is information available at baseline.
    # Exception (≤30 days after): keeps visits where the scan happened before weighing at the same visit
    # (weight does not change meaningfully within 30 days).
    m_prev = nearest(d, body_tbl, YEAR, direction="backward")
    m_post = nearest(d, body_tbl, 30.0, direction="forward")
    m = m_prev.merge(m_post[["pid", "t", "bmi"]], on=["pid", "t"],
                     how="left", suffixes=("", "_post"))
    m["bmi"] = m.bmi.fillna(m.bmi_post)
    d = d.merge(m[["pid", "t", "bmi"]], on=["pid", "t"])

    # ---- Step 5: sex ----
    d["sex"] = d.pid.map(sex_map)

    d = d.dropna(subset=["age", "sex", "bmi"])
    assert len(d) == d.pid.nunique()
    assert d.cohort.nunique() == 3, d.cohort.value_counts()

    # ---- ADNI only keeps ADNI1 / GO / 2 ----
    if adni_old_only:
        old = d.pid.map(adni_phase).isin(["ADNI1", "ADNIGO", "ADNI2"])
        d = d[(d.cohort != "ADNI") | old]

    # ---- Safety check: one row per participant ----
    d = d.sort_values("t").drop_duplicates("pid")
    return d


# Step 5a: age / sex / BMI
LB, IN = 0.45359237, 0.0254     # pound → kg, inch → m

def bmi_from(w_kg, h_m, pid):
    """medium of height --> BMI"""
    h_med = h_m.groupby(pid).transform("median")
    bmi = w_kg / h_med ** 2
    return bmi.where(bmi.between(12, 60))


def extra_oasis():
    S = ROOT / "OASIS3/OAS3_data_files/OASIS3_data_files/scans"
    f = lambda sub, name: next(S.glob(f"{sub}*/resources/csv/files/{name}"))
    c = pd.read_csv(f("UDSb4", "OASIS3_UDSb4_cdr.csv"))
    age = pd.DataFrame(dict(pid=c.OASISID, t=c.days_to_visit, age=c["age at visit"]))
    demo = pd.read_csv(f("demo-demographics", "OASIS3_demographics.csv"))
    sex = pd.Series(demo.GENDER.map({1: "M", 2: "F"}).values, index=demo.OASISID)
    b = pd.read_csv(f("UDSb1", "OASIS3_UDSb1_physical_eval.csv"))
    w = pd.to_numeric(b.WEIGHT, errors="coerce").where(lambda s: s.between(70, 450)) * LB
    h = pd.to_numeric(b.HEIGHT, errors="coerce").where(lambda s: s.between(48, 84)) * IN
    body = pd.DataFrame(dict(pid=b.OASISID, t=b.days_to_visit, bmi=bmi_from(w, h, b.OASISID)))
    return age, sex, body


def extra_adni():
    P = ROOT / "ADNI/ADNIMERGE2/ADNI_parquet"
    fs = pd.read_parquet(P / "ADSP_PHC_T1_FS.parquet", columns=["RID", "PHC_SCANDATE", "PHC_Age_T1"])
    age = pd.DataFrame(dict(pid=fs.RID, t=to_days(fs.PHC_SCANDATE), age=fs.PHC_Age_T1))
    dm = pd.read_parquet(P / "ADSP_PHC_DEMODX.parquet", columns=["RID", "PHC_Sex"]).dropna()
    sex = dm.groupby("RID").PHC_Sex.first().map({1: "M", 2: "F"})
    v = pd.read_parquet(P / "VITALS.parquet",
                        columns=["RID", "VISDATE", "VSWEIGHT", "VSWTUNIT", "VSHEIGHT", "VSHTUNIT"])

    def to_metric(x, unit, imp_word, met_word, factor):
        x = pd.to_numeric(x, errors="coerce").where(lambda s: s > 0)
        u = unit.astype(str).str.lower().str.strip()
        imp = u.str.contains(imp_word) | u.isin(["1", "1.0"])
        met = u.str.contains(met_word) | u.isin(["2", "2.0"])
        return pd.Series(np.where(imp, x * factor, np.where(met, x, np.nan)), index=x.index)

    w = to_metric(v.VSWEIGHT, v.VSWTUNIT, "pound", "kilo", LB).where(lambda s: s.between(30, 200))
    h = (to_metric(v.VSHEIGHT, v.VSHTUNIT, "inch", "cent", 2.54) / 100).where(lambda s: s.between(1.2, 2.1))
    body = pd.DataFrame(dict(pid=v.RID, t=to_days(v.VISDATE), bmi=bmi_from(w, h, v.RID)))
    return age, sex, body


def extra_nacc():
    u = pd.read_parquet(ROOT / "NACC/investigator_nacc75.parquet",
                        columns=["NACCID", "VISITDATE", "NACCAGE", "NACCSEX", "NACCBMI"])
    t = to_days(u.VISITDATE)
    age = pd.DataFrame(dict(pid=u.NACCID, t=t, age=u.NACCAGE))
    sex = u.groupby("NACCID").NACCSEX.first().map({1: "M", 2: "F"})
    body = pd.DataFrame(dict(pid=u.NACCID, t=t, bmi=u.NACCBMI.where(u.NACCBMI.between(12, 60))))
    return age, sex, body


ages, sexes, bodies = [], [], []
for name, fn in [("OASIS", extra_oasis), ("ADNI", extra_adni), ("NACC", extra_nacc)]:
    a, s, b = fn()
    for tbl in (a, b):
        tbl["pid"] = make_pid(name, tbl.pid)
    s.index = make_pid(name, s.index)
    ages.append(a); sexes.append(s); bodies.append(b)

age_tbl  = pd.concat(ages).dropna()
sex_map  = pd.concat(sexes).dropna()
body_tbl = pd.concat(bodies).dropna()

adni_phase = sob[sob.cohort == "ADNI"].groupby("pid").phase.first()

# print("age (number of people):", age_tbl.groupby(age_tbl.pid.str.split("_").str[0]).pid.nunique().to_dict())
# print("sex:", sex_map.value_counts().to_dict())
# print(body_tbl.bmi.describe().round(1))
# print("ADNI phase:", adni_phase.value_counts().to_dict())


# Step 6: stratified 5-fold
from sklearn.model_selection import StratifiedKFold
def make_folds(d, k=5, seed=42):
    delta = d.sob_fu - d.sob_bl
    age_bin = pd.cut(d.age, [0, 60, 70, 80, 200], right=False, labels=False)
    delta_bin = pd.cut(delta, [-np.inf, -0.01, 0, 1, 3, np.inf], labels=False)
    strata = d.sex + "_" + age_bin.astype(str) + "_" + delta_bin.astype(str)
    strata = strata.where(strata.map(strata.value_counts()) >= k, "rare")
    fold = np.empty(len(d), dtype=int)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for i, (_, te) in enumerate(skf.split(d, strata)):
        fold[te] = i
    return fold


d = build(adni_old_only=True, fu_from="cdr").reset_index(drop=True)
print("N =", len(d), d.groupby("cohort").size().to_dict())
d["fold"] = make_folds(d)
d.to_csv(OUT / "my_linear_null_dataset.csv", index=False)

# Step 7: Linear Null Model
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, median_absolute_error

NUM = ["age", "bmi", "sob_bl"]
CAT = ["sex"]


def make_model(num=NUM, cat=CAT):
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(drop="first"), cat),
    ])
    return Pipeline([("pre", pre), ("lr", LinearRegression())])


def metrics(y, p, y_train):
    mae = mean_absolute_error(y, p)
    return dict(MAE=mae,
                R2=r2_score(y, p),
                RMSE=np.sqrt(mean_squared_error(y, p)),
                MedAE=median_absolute_error(y, p),
                MASE=mae / np.mean(np.abs(y_train - y_train.mean())))

# Step 8: cross-validation
def run_cv(d, num, cat, label):
    rows, pred = [], pd.Series(np.nan, index=d.index)
    for k in range(d.fold.max() + 1):
        tr, te = d[d.fold != k], d[d.fold == k]
        p = make_model(num, cat).fit(tr[num + cat], tr.sob_fu).predict(te[num + cat])
        pred[te.index] = p
        rows.append(dict(model=label, fold=k + 1,
                         **metrics(te.sob_fu.values, p, tr.sob_fu.values)))
    return pd.DataFrame(rows), pred


res, _ = run_cv(d, NUM, CAT, "Linear Null")

M = ["MAE", "R2", "RMSE", "MedAE", "MASE"]
table2 = res.set_index("fold")[M].T
table2.columns = [f"Fold {k}" for k in table2.columns]
table2["Avg"] = table2.mean(axis=1)
print(table2.round(3))

paper = pd.DataFrame({"Fold 1": [0.980, 0.786], "Fold 2": [1.027, 0.804],
                      "Fold 3": [1.130, 0.757], "Fold 4": [0.998, 0.740],
                      "Fold 5": [1.012, 0.744]}, index=["MAE", "R2"])
paper["Avg"] = paper.mean(axis=1)
print("\nArticle Table 2:")
print(paper.round(3))

table2.round(3).to_csv(OUT / "my_table2_linear_null.csv")