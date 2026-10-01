-- 0006_add_sessions: server-side sessions backing the "Remember me" cookie.
--
-- The cookie used to carry a raw user id, which made it forgeable: writing
-- ``remember_me_logged_in_user=<any id>`` signed you in as that user, admin
-- included. It now carries an opaque random token whose SHA-256 lives here, so
-- possession can be revoked and guessing an id grants nothing.
--
--   * token_hash — SHA-256 of the token. The raw token only ever exists in the
--     browser cookie, so a leaked database can't be replayed against the app
--     (same one-way treatment as the invite tokens in 0003);
--   * user_id — whose session this is; deleting the user drops their sessions;
--   * expires_at — when the token stops being accepted;
--   * created_at — when it was issued.

CREATE TABLE sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Revoking every session of one user (a password change) looks up by user_id.
CREATE INDEX idx_sessions_user_id ON sessions(user_id);
