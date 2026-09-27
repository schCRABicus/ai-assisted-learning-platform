-- 0005_add_attempt_solution: the student's working, and manual score overrides.
--
-- A result row held only the student's final answer, which the admin could see
-- but not judge: nothing distinguished a slip from a genuine misunderstanding,
-- and an auto-grade that was wrong (or that deserved partial credit) could not
-- be corrected. Two columns close that gap:
--   * given_solution — the student's free-text working for the task, alongside
--     the answer. Optional: it stays NULL for a student who types only an
--     answer, and the auto-grader ignores it entirely.
--   * score_adjusted — whether an admin has manually overridden this row's
--     verdict and score. DEFAULT 0 means every pre-existing and every
--     auto-graded row reads as "not adjusted", so no backfill is needed.

ALTER TABLE attempt_results ADD COLUMN given_solution TEXT;

ALTER TABLE attempt_results ADD COLUMN score_adjusted INTEGER NOT NULL DEFAULT 0;
