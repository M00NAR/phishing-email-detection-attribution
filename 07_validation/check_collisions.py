"""
Collision-checking workflow for training_set.csv / holdout_set.csv.

"Collision" here covers four genuinely different failure modes, checked in order
from cheapest/most-mechanical to most-expensive/most-subtle. Each step is
independent -- read the printed report top to bottom; a later step failing
does not mean an earlier one was wrong.

  1. ID COLLISION         -- same `id` value used for rows that do NOT have the
                              same body text. Since `id` is an MD5 hash of the
                              body (truncated to 10 hex chars = 40 bits), this
                              would be a genuine hash collision. At this row
                              count (~13k total) the odds of a real one are
                              astronomically small (~13000^2 / 2 / 2^40 ≈ 4e-5),
                              so if this ever fires, suspect a pipeline bug
                              (e.g. id computed before some text-normalizing
                              step was applied to only some rows) before you
                              suspect MD5 itself.

  2. DUPLICATE CONTENT     -- same body text (possibly with a different id,
                              which shouldn't happen, or the same id, which is
                              the normal case) appearing more than once. This
                              is fine WITHIN a class (e.g. a phishing template
                              blasted to many people, already deduped once) but
                              is checked again here as a regression guard.

  3. CROSS-CLASS LABEL     -- the serious one: the exact same body text
     COLLISION                labeled differently in different rows (e.g. the
                              same email appearing as both "Legitimate" and
                              "Human-Phishing"). This is a direct contradiction
                              in the ground truth and would poison training.

  4. TRAIN/HOLDOUT OVERLAP -- any id (or, as a stronger check, any body text)
                              appearing in both training_set.csv and
                              holdout_set.csv. This would let the holdout set
                              leak into training and inflate reported accuracy.

  5. NEAR-DUPLICATE        -- not a strict collision, but worth surfacing:
     TEMPLATE REUSE            rows that are near-identical after stripping
                              whitespace/punctuation/digits (catches template
                              phishing emails that differ only in a name, date,
                              or amount, which exact-match dedup would miss).
                              Reported as a count, not a hard failure -- some
                              template reuse is expected and realistic for a
                              phishing corpus; the thing to watch for is
                              near-duplicate clusters that cross class labels.

Usage:
    python3 check_collisions.py /path/to/training_set.csv /path/to/holdout_set.csv
Defaults to the files in the same directory as this script if no args given.
"""
import sys, re, hashlib, json
from collections import defaultdict
import pandas as pd

def normalize_for_near_dup(s):
    s = str(s).lower()
    s = re.sub(r'\d+', '#', s)          # collapse all digits (dates, amounts, ids)
    s = re.sub(r'[^\w\s#]', '', s)      # strip punctuation
    s = re.sub(r'\s+', ' ', s).strip()  # collapse whitespace
    return s

def check_file(name, df):
    report = {"file": name, "rows": len(df)}

    # --- 1. ID collision: same id, different body ---
    id_to_bodies = defaultdict(set)
    for _id, body in zip(df["id"], df["body"].fillna("")):
        id_to_bodies[_id].add(body)
    true_hash_collisions = {k: v for k, v in id_to_bodies.items() if len(v) > 1}
    report["id_collisions_diff_content"] = len(true_hash_collisions)
    report["id_collision_examples"] = list(true_hash_collisions.keys())[:5]

    # --- 2. Duplicate content (same body, >1 row) ---
    body_counts = df["body"].fillna("").value_counts()
    dup_bodies = body_counts[body_counts > 1]
    report["duplicate_content_groups"] = int(len(dup_bodies))
    report["duplicate_content_rows"] = int(dup_bodies.sum())

    # --- 3. Cross-class label collision: same body, >1 distinct class ---
    body_to_classes = df.groupby(df["body"].fillna(""))["class"].agg(lambda s: set(s))
    label_conflicts = body_to_classes[body_to_classes.apply(len) > 1]
    report["cross_class_label_collisions"] = int(len(label_conflicts))
    if len(label_conflicts) > 0:
        report["cross_class_label_collision_examples"] = [
            {"body_snippet": b[:80], "classes": list(c)}
            for b, c in list(label_conflicts.items())[:5]
        ]

    # --- 5. Near-duplicate template reuse (within this file) ---
    norm = df["body"].fillna("").apply(normalize_for_near_dup)
    norm_counts = norm.value_counts()
    near_dup_groups = norm_counts[norm_counts > 1]
    report["near_duplicate_groups"] = int(len(near_dup_groups))
    report["near_duplicate_rows"] = int(near_dup_groups.sum())
    # does any near-dup cluster span more than one class?
    tmp = pd.DataFrame({"norm": norm, "class": df["class"]})
    cluster_classes = tmp[tmp["norm"].isin(near_dup_groups.index)].groupby("norm")["class"].agg(lambda s: set(s))
    cross_class_clusters = cluster_classes[cluster_classes.apply(len) > 1]
    report["near_duplicate_clusters_crossing_class"] = int(len(cross_class_clusters))

    return report, df

def main():
    args = sys.argv[1:]
    if len(args) == 2:
        train_path, holdout_path = args
    else:
        import os
        d = os.path.dirname(os.path.abspath(__file__))
        train_path = f"{d}/training_set.csv"
        holdout_path = f"{d}/holdout_set.csv"

    train = pd.read_csv(train_path)
    holdout = pd.read_csv(holdout_path)

    train_report, train = check_file("training_set.csv", train)
    holdout_report, holdout = check_file("holdout_set.csv", holdout)

    # --- 4. Train/holdout overlap ---
    overlap_ids = set(train["id"]) & set(holdout["id"])
    overlap_bodies = set(train["body"].fillna("")) & set(holdout["body"].fillna(""))
    cross_report = {
        "train_holdout_id_overlap": len(overlap_ids),
        "train_holdout_body_overlap": len(overlap_bodies),
    }

    full = {
        "training_set": train_report,
        "holdout_set": holdout_report,
        "cross_file": cross_report,
    }
    print(json.dumps(full, indent=2, default=str))

    # --- pass/fail summary against the checks that SHOULD always be zero ---
    hard_fails = []
    for r, name in [(train_report, "training_set.csv"), (holdout_report, "holdout_set.csv")]:
        if r["id_collisions_diff_content"] > 0:
            hard_fails.append(f"{name}: {r['id_collisions_diff_content']} true hash collisions")
        if r["cross_class_label_collisions"] > 0:
            hard_fails.append(f"{name}: {r['cross_class_label_collisions']} cross-class label collisions")
    if cross_report["train_holdout_id_overlap"] > 0:
        hard_fails.append(f"train/holdout id overlap: {cross_report['train_holdout_id_overlap']}")
    if cross_report["train_holdout_body_overlap"] > 0:
        hard_fails.append(f"train/holdout body overlap: {cross_report['train_holdout_body_overlap']}")

    print("\n" + "="*60)
    if hard_fails:
        print("HARD FAILURES (must be zero):")
        for f in hard_fails:
            print(" -", f)
        sys.exit(1)
    else:
        print("All hard-failure checks passed (0 id collisions, 0 cross-class label")
        print("collisions, 0 train/holdout overlap). Duplicate-content and")
        print("near-duplicate counts above are informational, not failures --")
        print("review them manually, especially any 'crossing_class' count > 0.")

if __name__ == "__main__":
    main()
