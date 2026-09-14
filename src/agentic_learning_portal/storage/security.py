"""Password hashing for the portal's local storage.

Uses ``hashlib.scrypt`` (memory-hard, standard-library) with a random per-hash
salt so identical passwords never produce identical stored values. A stored hash
is the self-describing string::

    scrypt$<n>$<r>$<p>$<salt_b64>$<hash_b64>
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

_SCRYPT_N = 2**14  # 16 MiB of memory at r=8
_SCRYPT_R = 8
_SCRYPT_P = 1
_SALT_BYTES = 16


def hash_password(password: str) -> str:
    """Return a salted scrypt hash for ``password``."""
    salt = secrets.token_bytes(_SALT_BYTES)
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
    )
    return "scrypt${}${}${}${}${}".format(
        _SCRYPT_N,
        _SCRYPT_R,
        _SCRYPT_P,
        base64.b64encode(salt).decode(),
        base64.b64encode(derived).decode(),
    )


def verify_password(password: str, stored: str) -> bool:
    """Return whether ``password`` matches ``stored`` (a ``hash_password`` value)."""
    try:
        scheme, n, r, p, salt_b64, hash_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        derived = hashlib.scrypt(
            password.encode("utf-8"),
            salt=base64.b64decode(salt_b64),
            n=int(n),
            r=int(r),
            p=int(p),
        )
        expected = base64.b64decode(hash_b64)
        return hmac.compare_digest(derived, expected)
    except (ValueError, TypeError):
        return False


def hash_token(token: str) -> str:
    """Return a one-way SHA-256 digest of an invite/verification token.

    Unlike passwords, verification tokens are high-entropy random strings
    (``secrets.token_urlsafe``), so a bare SHA-256 digest is enough — no salt
    or key-stretching needed. The raw token only ever appears in the emailed
    link; storage keeps this digest.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def verify_token(token: str, stored: str) -> bool:
    """Return whether ``token`` hashes to ``stored`` (a ``hash_token`` value)."""
    return hmac.compare_digest(hash_token(token), stored)