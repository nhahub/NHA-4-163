"""Envelope encryption for PHI fields stored at rest.

Design
------
Uses the Fernet symmetric scheme (AES-128-CBC + HMAC-SHA256) from the
``cryptography`` package.  The master key (KEK — Key Encryption Key) is
loaded from the ``ENCRYPTION_KEY`` environment variable.

Envelope format
---------------
Ciphertext is stored as ``{key_id}:{fernet_token}`` so the decryption
path can select the correct key when multiple key versions are active
during rotation.

Key rotation
------------
Set ``ENCRYPTION_PREVIOUS_KEY`` to the old key while ``ENCRYPTION_KEY``
holds the new one.  ``decrypt()`` will try the new key first, then fall
back to the old key transparently.  Re-encrypt stored values at your
convenience; remove the old key once all rows are migrated.

Usage::

    svc = get_encryption_service()
    ct = svc.encrypt("123-45-6789")   # SSN
    pt = svc.decrypt(ct)              # "123-45-6789"

PHI note: keys are loaded from environment variables — they are NEVER
logged, stored in source code, or included in API responses.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

log = logging.getLogger(__name__)

_SEPARATOR = ":"
_DEFAULT_KEY_ID = "v1"


class EncryptionError(Exception):
    """Raised when encryption or decryption fails."""


class EncryptionService:
    """Fernet-based symmetric encryption with key-rotation support.

    Attributes:
        key_id: Identifier written alongside every ciphertext token.
    """

    def __init__(
        self,
        primary_key_b64: str,
        key_id: str = _DEFAULT_KEY_ID,
        previous_key_b64: str | None = None,
    ) -> None:
        """Initialise the encryption service.

        Args:
            primary_key_b64: URL-safe base64-encoded 32-byte Fernet key.
                Generate with ``Fernet.generate_key().decode()``.
            key_id: Identifier stored alongside ciphertext for key
                selection during rotation.
            previous_key_b64: Optional previous Fernet key used for
                transparent decryption of older ciphertext during rotation.

        Raises:
            EncryptionError: If any key is malformed.
        """
        from cryptography.fernet import Fernet, InvalidToken, MultiFernet

        try:
            self._primary = Fernet(primary_key_b64.encode())
        except Exception as exc:
            raise EncryptionError(f"Invalid primary encryption key: {exc}") from exc

        ferners = [self._primary]
        if previous_key_b64:
            try:
                ferners.append(Fernet(previous_key_b64.encode()))
            except Exception as exc:
                raise EncryptionError(f"Invalid previous encryption key: {exc}") from exc

        self._multi = MultiFernet(ferners)
        self._InvalidToken = InvalidToken
        self.key_id = key_id

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a UTF-8 string and return an envelope-encoded token.

        Args:
            plaintext: String to encrypt (e.g., a PHI field value).

        Returns:
            Envelope string in the form ``{key_id}:{fernet_token}``.

        Raises:
            EncryptionError: On unexpected encryption failure.
        """
        try:
            token = self._primary.encrypt(plaintext.encode("utf-8"))
            return f"{self.key_id}{_SEPARATOR}{token.decode('ascii')}"
        except Exception as exc:
            raise EncryptionError("Encryption failed") from exc

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt an envelope-encoded token.

        Tries the primary key first, then the previous key (if configured).

        Args:
            ciphertext: Envelope string returned by ``encrypt()``.

        Returns:
            Original plaintext string.

        Raises:
            EncryptionError: If the token is malformed or cannot be
                decrypted with any loaded key.
        """
        try:
            # Strip key_id prefix — everything after the first separator is the Fernet token
            parts = ciphertext.split(_SEPARATOR, 1)
            if len(parts) != 2:
                raise EncryptionError("Malformed ciphertext: missing key_id prefix")
            token = parts[1].encode("ascii")
            return self._multi.decrypt(token).decode("utf-8")
        except self._InvalidToken as exc:
            raise EncryptionError("Decryption failed — token invalid or key mismatch") from exc
        except EncryptionError:
            raise
        except Exception as exc:
            raise EncryptionError("Decryption failed") from exc

    def rotate(self, ciphertext: str) -> str:
        """Re-encrypt a ciphertext under the current primary key.

        Used during key rotation to migrate old ciphertext to the new key.

        Args:
            ciphertext: Existing envelope-encoded ciphertext.

        Returns:
            New envelope ciphertext encrypted under the primary key.

        Raises:
            EncryptionError: If decryption of the old token fails.
        """
        return self.encrypt(self.decrypt(ciphertext))

    @staticmethod
    def generate_key() -> str:
        """Return a new URL-safe base64-encoded Fernet key string.

        Returns:
            32-byte Fernet key encoded as URL-safe base64.
        """
        from cryptography.fernet import Fernet

        return Fernet.generate_key().decode("ascii")


@lru_cache(maxsize=1)
def get_encryption_service() -> EncryptionService:
    """Return the application-scoped EncryptionService singleton.

    Reads ``ENCRYPTION_KEY``, ``ENCRYPTION_KEY_ID``, and optionally
    ``ENCRYPTION_PREVIOUS_KEY`` from the environment.

    Returns:
        Configured EncryptionService.

    Raises:
        EncryptionError: If ``ENCRYPTION_KEY`` is missing or invalid.
    """
    key = os.environ.get("ENCRYPTION_KEY", "")
    if not key:
        raise EncryptionError(
            "ENCRYPTION_KEY environment variable is not set. "
            'Generate one with: python -c "from cryptography.fernet import Fernet; '
            'print(Fernet.generate_key().decode())"'
        )
    key_id = os.environ.get("ENCRYPTION_KEY_ID", _DEFAULT_KEY_ID)
    previous = os.environ.get("ENCRYPTION_PREVIOUS_KEY") or None
    return EncryptionService(key, key_id=key_id, previous_key_b64=previous)


def encrypt_field(plaintext: str | None) -> str | None:
    """Convenience wrapper — encrypt a nullable string field.

    Args:
        plaintext: Value to encrypt, or ``None`` (returned as-is).

    Returns:
        Encrypted envelope string, or ``None`` if input was ``None``.
    """
    if plaintext is None:
        return None
    return get_encryption_service().encrypt(plaintext)


def decrypt_field(ciphertext: str | None) -> str | None:
    """Convenience wrapper — decrypt a nullable ciphertext field.

    Args:
        ciphertext: Envelope string to decrypt, or ``None``.

    Returns:
        Decrypted plaintext, or ``None`` if input was ``None``.
    """
    if ciphertext is None:
        return None
    return get_encryption_service().decrypt(ciphertext)
