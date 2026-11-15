"""
student_vectors.py
==================
Two vector representations of students, built only from course enrollments,
stored in one Qdrant vector database:

  1. "friends" vector  -> students who chose the same groups (sections)
                          in the big multi-group courses end up close.
                          Method: Multiple Correspondence Analysis (MCA).
  2. "field"   vector  -> 7D field-of-study attitude
                          (Communications, RF, Power, Electronics, Bio, Digital, Control).
                          Method: smoothed course->field priors, log-lift sum
                          (updated like a GPA), softmax, then sqrt -> unit sphere.

Input files (CSV)
-----------------
enrollments.csv : student, course, term, group, credits
                  term must be sortable, e.g. 14021, 14022 (year + semester)
fields.csv      : student, field, declared_term
                  only students who have already chosen their field

Usage
-----
python student_vectors.py --enrollments enrollments.csv --fields fields.csv --db ./student_db
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

FIELDS = ["Communications", "RF", "Power", "Electronics", "Bio", "Digital", "Control"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_data(enroll_path: str, fields_path: str | None = None):
    enr = pd.read_csv(enroll_path)
    need = {"student", "course", "term", "group"}
    missing = need - set(enr.columns)
    if missing:
        raise ValueError(f"enrollments file is missing columns: {missing}")
    if "credits" not in enr.columns:
        enr["credits"] = 3
    enr["student"] = enr["student"].astype(str)

    fields = None
    if fields_path:
        fields = pd.read_csv(fields_path)
        fields["student"] = fields["student"].astype(str)
        bad = set(fields["field"]) - set(FIELDS)
        if bad:
            raise ValueError(f"unknown field names: {bad}; expected {FIELDS}")
    return enr, fields


# ---------------------------------------------------------------------------
# 1) FRIENDS vectors
# ---------------------------------------------------------------------------
def select_courses(enr: pd.DataFrame, min_groups: int = 5, min_coverage: float = 0.5):
    """Courses with many groups per term (a real choice) that most students take."""
    groups_per_term = enr.groupby(["course", "term"])["group"].nunique()
    avg_groups = groups_per_term.groupby("course").mean()
    coverage = enr.groupby("course")["student"].nunique() / enr["student"].nunique()
    chosen = avg_groups[(avg_groups >= min_groups) & (coverage.reindex(avg_groups.index) >= min_coverage)]
    return sorted(chosen.index)


def build_wide_table(enr: pd.DataFrame, courses: list[str]) -> pd.DataFrame:
    """One row per student, one column per course, value = 'term_group' or 'NULL'."""
    e = enr[enr["course"].isin(courses)].copy()
    e["term_group"] = e["term"].astype(str) + "_g" + e["group"].astype(str)
    e = e.sort_values("term")  # 'last' keeps the latest attempt for retakes
    wide = e.pivot_table(index="student", columns="course", values="term_group", aggfunc="last")
    all_students = sorted(enr["student"].unique())
    return wide.reindex(all_students).fillna("NULL")


def friend_vectors(wide: pd.DataFrame, k: int = 16, use_mca: bool = True) -> pd.DataFrame:
    """MCA on the categorical table -> k-dim vector per student (unit length)."""
    k = min(k, max(2, wide.shape[1] * 3))
    coords = None
    if use_mca:
        try:
            import prince
            mca = prince.MCA(n_components=k, random_state=0).fit(wide)
            coords = mca.row_coordinates(wide)
        except Exception as ex:  # fall back to plain SVD if prince is missing / fails
            print(f"[friends] MCA failed ({ex}); falling back to one-hot + SVD")
    if coords is None:
        from sklearn.decomposition import TruncatedSVD
        onehot = pd.get_dummies(wide)
        onehot = onehot.loc[:, ~onehot.columns.str.endswith("_NULL")]  # NULL carries no info here
        coords = pd.DataFrame(TruncatedSVD(k, random_state=0).fit_transform(onehot.values),
                              index=wide.index)
    X = coords.values.astype(float)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return pd.DataFrame(X, index=wide.index)


# ---------------------------------------------------------------------------
# 2) FIELD vectors
# ---------------------------------------------------------------------------
class FieldModel:
    """
    p_c(f)   = (n_cf + m*pi_f) / (n_c + m)       smoothed share of course c's students in field f
    lift_c   = log(p_c / pi)                      ~0 for mandatory courses, big for telling electives
    score(f) = log pi_f + alpha * sum_c w_c * lift_c(f)   (w_c = credits/3, updated like a GPA)
    P        = softmax(score)
    vector   = sqrt(P)                            lies on the positive part of the 7D unit sphere
    """

    def __init__(self, m: float = 20.0, alpha: float = 0.5):
        self.m, self.alpha = m, alpha

    def fit(self, enr: pd.DataFrame, fields: pd.DataFrame, upto_term=None):
        f = fields if upto_term is None else fields[fields["declared_term"] <= upto_term]
        declared = f.set_index("student")["field"]
        e = enr[enr["student"].isin(declared.index)].drop_duplicates(["student", "course"])
        e = e.assign(field=e["student"].map(declared))

        field_counts = declared.value_counts().reindex(FIELDS, fill_value=0).values.astype(float)
        self.pi = (field_counts + 1) / (field_counts.sum() + len(FIELDS))  # Laplace-smoothed base rate

        ct = pd.crosstab(e["course"], e["field"]).reindex(columns=FIELDS, fill_value=0)
        n_c = ct.sum(axis=1).values[:, None]
        p = (ct.values + self.m * self.pi) / (n_c + self.m)
        self.course_p = pd.DataFrame(p, index=ct.index, columns=FIELDS)
        self.lift = pd.DataFrame(np.log(p / self.pi), index=ct.index, columns=FIELDS)
        return self

    def scores(self, enr: pd.DataFrame, students=None) -> pd.DataFrame:
        e = enr.drop_duplicates(["student", "course"])
        if students is not None:
            e = e[e["student"].isin(students)]
        e = e[e["course"].isin(self.lift.index)]          # unseen courses add nothing
        w = (e["credits"].values / 3.0)[:, None]
        contrib = pd.DataFrame(w * self.lift.loc[e["course"]].values,
                               index=e["student"].values, columns=FIELDS)
        s = contrib.groupby(level=0).sum() * self.alpha + np.log(self.pi)
        all_s = sorted(set(enr["student"]) if students is None else set(students))
        return s.reindex(all_s).fillna(pd.Series(np.log(self.pi), index=FIELDS))

    def probabilities(self, enr, students=None) -> pd.DataFrame:
        s = self.scores(enr, students)
        z = np.exp(s.values - s.values.max(axis=1, keepdims=True))
        return pd.DataFrame(z / z.sum(1, keepdims=True), index=s.index, columns=FIELDS)

    def vectors(self, enr, students=None) -> pd.DataFrame:
        return np.sqrt(self.probabilities(enr, students))  # unit length by construction


def update_student(score: np.ndarray, model: FieldModel, course: str, credits: float) -> np.ndarray:
    """GPA-style incremental update when a student finishes one more course."""
    if course in model.lift.index:
        score = score + model.alpha * (credits / 3.0) * model.lift.loc[course].values
    return score


def evaluate_fields(enr, fields, test_term, m=20.0, alpha=0.5):
    """Temporal hold-out: learn from students declared before test_term, predict those declared at it."""
    model = FieldModel(m, alpha).fit(enr, fields[fields["declared_term"] < test_term])
    test = fields[fields["declared_term"] == test_term]
    if test.empty:
        return None
    # only use courses the test students took before declaring
    e_test = enr[enr["student"].isin(test["student"])]
    e_test = e_test.merge(test[["student", "declared_term"]], on="student")
    e_test = e_test[e_test["term"] < e_test["declared_term"]]
    P = model.probabilities(e_test, students=test["student"])
    truth = test.set_index("student")["field"].reindex(P.index)
    pred = P.idxmax(axis=1)
    top2 = P.apply(lambda r: list(r.nlargest(2).index), axis=1)
    p_true = P.values[np.arange(len(P)), [FIELDS.index(t) for t in truth]]
    return {
        "test_term": test_term,
        "n": len(P),
        "accuracy": float((pred == truth).mean()),
        "top2": float(np.mean([t in tt for t, tt in zip(truth, top2)])),
        "log_loss": float(-np.log(p_true + 1e-12).mean()),
    }


def tune_alpha(enr, fields, test_term, grid=(0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)):
    results = [(a, evaluate_fields(enr, fields, test_term, alpha=a)) for a in grid]
    results = [(a, r) for a, r in results if r]
    best = min(results, key=lambda ar: ar[1]["log_loss"])
    return best[0], results


# ---------------------------------------------------------------------------
# 3) Vector database (Qdrant, two named vectors per student)
# ---------------------------------------------------------------------------
def build_db(friends: pd.DataFrame, field_vecs: pd.DataFrame, path: str,
             declared: pd.Series | None = None, collection: str = "students"):
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct

    client = QdrantClient(path=path)
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(collection, vectors_config={
        "friends": VectorParams(size=friends.shape[1], distance=Distance.COSINE),
        "field": VectorParams(size=len(FIELDS), distance=Distance.COSINE),
    })
    students = friends.index.intersection(field_vecs.index)
    points = []
    for i, sid in enumerate(students):
        P = field_vecs.loc[sid].values ** 2
        payload = {"student": sid,
                   "predicted_field": FIELDS[int(P.argmax())],
                   "field_probs": dict(zip(FIELDS, np.round(P, 4).tolist()))}
        if declared is not None and sid in declared.index:
            payload["declared_field"] = declared[sid]
        points.append(PointStruct(id=i, payload=payload, vector={
            "friends": friends.loc[sid].tolist(),
            "field": field_vecs.loc[sid].tolist(),
        }))
    for j in range(0, len(points), 512):
        client.upsert(collection, points=points[j:j + 512])
    return client


def _point_of(client, student, collection="students"):
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    pts, _ = client.scroll(collection, with_vectors=True, limit=1, scroll_filter=Filter(
        must=[FieldCondition(key="student", match=MatchValue(value=str(student)))]))
    if not pts:
        raise KeyError(f"student {student} not in database")
    return pts[0]


def find_similar(client, student, using="friends", k=10, collection="students"):
    """using='friends' -> likely friends; using='field' -> students with the most similar field attitude."""
    p = _point_of(client, student, collection)
    hits = client.query_points(collection, query=p.vector[using], using=using,
                               limit=k + 1, with_payload=True).points
    return [(h.payload["student"], round(h.score, 4)) for h in hits if h.id != p.id][:k]


def cluster_friends(friends: pd.DataFrame, min_cluster_size: int = 3) -> pd.Series:
    """HDBSCAN on the friend vectors. Label -1 = no clear group."""
    from sklearn.cluster import HDBSCAN
    labels = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", copy=True).fit_predict(friends.values)
    return pd.Series(labels, index=friends.index, name="friend_group")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enrollments", required=True)
    ap.add_argument("--fields", required=True)
    ap.add_argument("--db", default="./student_db")
    ap.add_argument("--k", type=int, default=16, help="friend vector dimension")
    ap.add_argument("--min-groups", type=int, default=5)
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    enr, fields = load_data(args.enrollments, args.fields)

    # ---- friends
    courses = select_courses(enr, args.min_groups, args.min_coverage)
    print(f"[friends] {len(courses)} multi-group courses selected: {courses}")
    wide = build_wide_table(enr, courses)
    F = friend_vectors(wide, args.k)
    groups = cluster_friends(F)
    print(f"[friends] {groups.max() + 1} groups found, {int((groups == -1).sum())} students unassigned")

    # ---- field: tune alpha on the latest declaration term, then fit on everything
    last_term = fields["declared_term"].max()
    alpha, res = tune_alpha(enr, fields, last_term)
    for a, r in res:
        print(f"[field] alpha={a:<5} acc={r['accuracy']:.3f} top2={r['top2']:.3f} logloss={r['log_loss']:.3f}")
    print(f"[field] chosen alpha = {alpha}")
    model = FieldModel(alpha=alpha).fit(enr, fields)
    V = model.vectors(enr)

    # ---- database
    declared = fields.set_index("student")["field"]
    client = build_db(F, V, args.db, declared)
    print(f"[db] stored {client.count('students').count} students in {args.db}")

    # ---- export a flat table
    P = V ** 2
    out = pd.concat([groups, P.add_prefix("p_"), P.idxmax(axis=1).rename("predicted_field"),
                     declared.rename("declared_field")], axis=1)
    out.index.name = "student"
    out.to_csv(args.out)
    print(f"[out] wrote {args.out}")


if __name__ == "__main__":
    main()
