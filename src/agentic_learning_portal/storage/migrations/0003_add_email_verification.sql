-- 0003_add_email_verification: track invite-based account verification.
--
-- A user invited by an admin (created without a password) can't sign in until
-- they open the emailed verification link and set a password. This adds:
--   * email_verified — whether the email was confirmed (defaults to 1 so the
--     seeded admin and any user created with a password is immediately usable);
--   * verification_token_hash — SHA-256 of the one-time invite token (the raw
--     token only ever appears in the emailed link, never here);
--   * verification_expires_at — when the pending token stops being accepted.

ALTER TABLE users ADD COLUMN email_verified INTEGER NOT NULL DEFAULT 1;
ALTER TABLE users ADD COLUMN verification_token_hash TEXT;
ALTER TABLE users ADD COLUMN verification_expires_at TEXT;