DROP INDEX IF EXISTS idx_attempt_results_attempt_task;

ALTER TABLE attempts DROP COLUMN results_seen;

ALTER TABLE assignments DROP COLUMN extra_attempts;
