-- 0004_add_attempt_tracking: retake entitlement and result notifications.
--
-- Taking an assignment is a two-step flow now: the student saves progress into
-- the attempt's results while it is ``in_progress``, then submits and the
-- attempt is graded and flipped to ``completed``. That needs:
--   * extra_attempts — how many *additional* attempts the admin has granted on
--     top of the student's first one. A student may hold at most
--     ``1 + extra_attempts`` attempts; granting another just increments this
--     (nothing is decremented, so the entitlement can't drift).
--   * results_seen — whether the admin has looked at a submitted, graded
--     attempt. It drives the "new results" panel, so submissions start unseen.
--   * one row per (attempt, task): saving progress and grading both upsert into
--     the same row instead of appending a duplicate every save.

ALTER TABLE assignments ADD COLUMN extra_attempts INTEGER NOT NULL DEFAULT 0;

ALTER TABLE attempts ADD COLUMN results_seen INTEGER NOT NULL DEFAULT 0;

-- Collapse any pre-existing duplicates (keep the newest row per pair) before
-- the unique index makes them impossible.
DELETE FROM attempt_results
 WHERE id NOT IN (
     SELECT MAX(id) FROM attempt_results GROUP BY attempt_id, task_id
 );

CREATE UNIQUE INDEX idx_attempt_results_attempt_task
    ON attempt_results (attempt_id, task_id);
