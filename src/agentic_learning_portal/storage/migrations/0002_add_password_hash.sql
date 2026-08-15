-- 0002_add_password_hash: allow storing a password hash for authentication.
--
-- The initial admin seeded from ADMIN_USERNAME/ADMIN_PASSWORD (and any future
-- user) stores a salted scrypt hash here, never the plaintext password.

ALTER TABLE users ADD COLUMN password_hash TEXT;