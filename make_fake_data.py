"""
Generate fake enrollment data with KNOWN friend groups and fields,
so the pipeline in student_vectors.py can be tested end to end.

Writes: enrollments.csv, fields.csv, truth.csv  (truth = real friend group + real field)
"""
import numpy as np
import pandas as pd

from student_vectors import FIELDS

rng = np.random.default_rng(0)

ENTRY_YEARS = [1400, 1401, 1402, 1403]
STUDENTS_PER_YEAR = 150
CURRENT_TERM_INDEX = lambda year: (1404 - year) * 2 + 1   # semesters completed up to 1404-1


def term_of(entry_year, sem_index):            # sem_index 0 = first semester
    return (entry_year + sem_index // 2) * 10 + (sem_index % 2) + 1


# multi-group mandatory courses: (name, semester index, number of groups)
BIG_COURSES = [("Circuits1", 1, 6), ("EngMath", 2, 6), ("Signals", 3, 6), ("Electronics1", 3, 5),
               ("Circuits2", 2, 5), ("Electromagnetics", 4, 6), ("Probability", 3, 5), ("LinearControl", 5, 5)]
# single-group mandatory courses (no information)
SMALL_COURSES = [("Physics1", 0), ("Calculus1", 0), ("Physics2", 1), ("Calculus2", 1), ("Programming", 0)]
# field electives: 3 per field, taken in semesters 5..7
ELECTIVES = {f: [f"{f[:4]}_Elective{i}" for i in range(1, 4)] for f in FIELDS}
DECLARE_SEM = 6

rows, fields_rows, truth_rows = [], [], []
sid = 0
for year in ENTRY_YEARS:
    done = CURRENT_TERM_INDEX(year)
    students = []
    # split the cohort into friend groups of 3..5
    remaining = STUDENTS_PER_YEAR
    gid = 0
    while remaining > 0:
        size = min(remaining, rng.integers(3, 6))
        group_field = rng.choice(FIELDS)
        for _ in range(size):
            field = group_field if rng.random() < 0.5 else rng.choice(FIELDS)
            students.append((f"S{sid:04d}", f"{year}-{gid}", field))
            sid += 1
        remaining -= size
        gid += 1

    friend_groups = {}
    for s, g, f in students:
        friend_groups.setdefault(g, []).append((s, f))

    for g, members in friend_groups.items():
        # each friend group agrees on a group number per big course
        chosen = {c: rng.integers(1, n + 1) for c, _, n in BIG_COURSES}
        for s, field in members:
            truth_rows.append((s, g, field, year))
            for c, sem in SMALL_COURSES:
                if sem < done:
                    rows.append((s, c, term_of(year, sem), 1, 3))
            for c, sem, n in BIG_COURSES:
                if sem < done:
                    grp = chosen[c] if rng.random() < 0.8 else rng.integers(1, n + 1)
                    rows.append((s, c, term_of(year, sem), grp, 3))
            for f2, courses in ELECTIVES.items():
                for c in courses:
                    p = 0.6 if f2 == field else 0.06
                    if rng.random() < p:
                        sem = rng.integers(5, 8)
                        if sem < done:
                            rows.append((s, c, term_of(year, sem), 1, 3))
            if DECLARE_SEM < done:
                fields_rows.append((s, field, term_of(year, DECLARE_SEM)))

pd.DataFrame(rows, columns=["student", "course", "term", "group", "credits"]).to_csv("enrollments.csv", index=False)
pd.DataFrame(fields_rows, columns=["student", "field", "declared_term"]).to_csv("fields.csv", index=False)
pd.DataFrame(truth_rows, columns=["student", "true_group", "true_field", "entry_year"]).to_csv("truth.csv", index=False)
print(f"{sid} students, {len(rows)} enrollments, {len(fields_rows)} declared fields")
