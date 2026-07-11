"""Unit tests for Phase 7 — Security & Compliance.

Covers:
- libs/common/encryption.py  — EncryptionService round-trips, key rotation,
                               error cases
- libs/common/deidentification.py — HIPAA Safe Harbor generalisation rules
- services/api/auth/jwt.py   — token creation, verification, expiry
- services/api/auth/rbac.py  — permission matrix, unknown role rejection
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import date
from typing import Any
from unittest.mock import patch

import pytest

# ── helpers ────────────────────────────────────────────────────────────────────


def _fresh_fernet_key() -> str:
    from cryptography.fernet import Fernet

    return Fernet.generate_key().decode()


# ── EncryptionService ─────────────────────────────────────────────────────────


class TestEncryptionService:
    def test_round_trip(self) -> None:
        from libs.common.encryption import EncryptionService

        svc = EncryptionService(_fresh_fernet_key())
        ct = svc.encrypt("123-45-6789")
        assert svc.decrypt(ct) == "123-45-6789"

    def test_ciphertext_not_plaintext(self) -> None:
        from libs.common.encryption import EncryptionService

        svc = EncryptionService(_fresh_fernet_key())
        ct = svc.encrypt("secret")
        assert "secret" not in ct

    def test_key_id_prefix(self) -> None:
        from libs.common.encryption import EncryptionService

        svc = EncryptionService(_fresh_fernet_key(), key_id="prod-v2")
        ct = svc.encrypt("data")
        assert ct.startswith("prod-v2:")

    def test_different_plaintexts_produce_different_ciphertexts(self) -> None:
        from libs.common.encryption import EncryptionService

        svc = EncryptionService(_fresh_fernet_key())
        assert svc.encrypt("aaa") != svc.encrypt("bbb")

    def test_same_plaintext_produces_different_ciphertexts(self) -> None:
        # Fernet uses random IV — same input should not produce same output
        from libs.common.encryption import EncryptionService

        svc = EncryptionService(_fresh_fernet_key())
        assert svc.encrypt("same") != svc.encrypt("same")

    def test_wrong_key_raises(self) -> None:
        from libs.common.encryption import EncryptionError, EncryptionService

        svc_a = EncryptionService(_fresh_fernet_key())
        svc_b = EncryptionService(_fresh_fernet_key())
        ct = svc_a.encrypt("hello")
        with pytest.raises(EncryptionError):
            svc_b.decrypt(ct)

    def test_malformed_ciphertext_raises(self) -> None:
        from libs.common.encryption import EncryptionError, EncryptionService

        svc = EncryptionService(_fresh_fernet_key())
        with pytest.raises(EncryptionError):
            svc.decrypt("no-separator-here")

    def test_key_rotation_transparent_decrypt(self) -> None:
        from libs.common.encryption import EncryptionService

        old_key = _fresh_fernet_key()
        new_key = _fresh_fernet_key()
        # Encrypt with old key
        old_svc = EncryptionService(old_key, key_id="v1")
        ct = old_svc.encrypt("phi-data")
        # New service with new primary + old previous key can still decrypt
        new_svc = EncryptionService(new_key, key_id="v2", previous_key_b64=old_key)
        assert new_svc.decrypt(ct) == "phi-data"

    def test_rotate_re_encrypts_under_new_key(self) -> None:
        from libs.common.encryption import EncryptionService

        old_key = _fresh_fernet_key()
        new_key = _fresh_fernet_key()
        old_svc = EncryptionService(old_key, key_id="v1")
        ct_old = old_svc.encrypt("value")
        new_svc = EncryptionService(new_key, key_id="v2", previous_key_b64=old_key)
        ct_new = new_svc.rotate(ct_old)
        assert ct_new.startswith("v2:")
        # New service without old key can now decrypt
        clean_svc = EncryptionService(new_key, key_id="v2")
        assert clean_svc.decrypt(ct_new) == "value"

    def test_generate_key_returns_valid_fernet_key(self) -> None:
        from cryptography.fernet import Fernet

        from libs.common.encryption import EncryptionService

        key = EncryptionService.generate_key()
        # Should not raise
        Fernet(key.encode())

    def test_encrypt_decrypt_none_convenience(self) -> None:
        from libs.common.encryption import decrypt_field, encrypt_field

        key = _fresh_fernet_key()
        with patch.dict(os.environ, {"ENCRYPTION_KEY": key, "ENCRYPTION_KEY_ID": "t1"}):
            from libs.common import encryption

            encryption.get_encryption_service.cache_clear()
            assert encrypt_field(None) is None
            assert decrypt_field(None) is None
            ct = encrypt_field("test-value")
            assert ct is not None
            assert decrypt_field(ct) == "test-value"
            encryption.get_encryption_service.cache_clear()

    def test_invalid_key_raises_encryption_error(self) -> None:
        from libs.common.encryption import EncryptionError, EncryptionService

        with pytest.raises(EncryptionError):
            EncryptionService("not-a-valid-fernet-key")


# ── HIPAA De-identification ────────────────────────────────────────────────────


class TestDeidentification:
    def test_name_fields_redacted(self) -> None:
        from libs.common.deidentification import deidentify_patient

        result = deidentify_patient({"first_name": "Alice", "last_name": "Smith"})
        assert result["first_name"] == "[REDACTED]"
        assert result["last_name"] == "[REDACTED]"

    def test_non_phi_fields_preserved(self) -> None:
        from libs.common.deidentification import deidentify_patient

        result = deidentify_patient({"gender": "female", "comorbidity_count": 3})
        assert result["gender"] == "female"
        assert result["comorbidity_count"] == 3

    def test_zip_code_generalised(self) -> None:
        from libs.common.deidentification import deidentify_patient

        result = deidentify_patient({"zip_code": "10001"})
        assert result["zip_code"] == "100**"

    def test_restricted_zip_replaced_with_000(self) -> None:
        from libs.common.deidentification import deidentify_patient

        result = deidentify_patient({"zip_code": "036XX"})
        assert result["zip_code"] == "000**"

    def test_age_under_90_returns_decade(self) -> None:
        from libs.common.deidentification import generalise_age

        assert generalise_age(35) == "30s"
        assert generalise_age(0) == "0s"
        assert generalise_age(89) == "80s"

    def test_age_90_plus_returns_sentinel(self) -> None:
        from libs.common.deidentification import generalise_age

        assert generalise_age(90) == "90+"
        assert generalise_age(110) == "90+"

    def test_age_none_returns_none(self) -> None:
        from libs.common.deidentification import generalise_age

        assert generalise_age(None) is None

    def test_date_generalised_to_year_month(self) -> None:
        from libs.common.deidentification import generalise_date

        result = generalise_date("1980-06-15", age=44)
        assert result == "1980-06"

    def test_date_generalised_to_year_only_for_90_plus(self) -> None:
        from libs.common.deidentification import generalise_date

        result = generalise_date("1925-03-12", age=101)
        assert result == "1925"

    def test_date_accepts_date_object(self) -> None:
        from libs.common.deidentification import generalise_date

        result = generalise_date(date(1975, 8, 20), age=50)
        assert result == "1975-08"

    def test_date_none_returns_none(self) -> None:
        from libs.common.deidentification import generalise_date

        assert generalise_date(None) is None

    def test_ssn_removed(self) -> None:
        from libs.common.deidentification import deidentify_patient

        result = deidentify_patient({"ssn": "123-45-6789"})
        assert result["ssn"] == "[REDACTED]"

    def test_email_removed(self) -> None:
        from libs.common.deidentification import deidentify_patient

        result = deidentify_patient({"email": "patient@example.com"})
        assert result["email"] == "[REDACTED]"

    def test_ip_address_removed(self) -> None:
        from libs.common.deidentification import deidentify_patient

        result = deidentify_patient({"ip_address": "192.168.1.1"})
        assert result["ip_address"] == "[REDACTED]"

    def test_nested_dict_recursion(self) -> None:
        from libs.common.deidentification import deidentify_patient

        record: dict[str, Any] = {"metadata": {"email": "x@y.com", "score": 0.8}}
        result = deidentify_patient(record)
        assert result["metadata"]["email"] == "[REDACTED]"
        assert result["metadata"]["score"] == 0.8

    def test_list_of_dicts_recursion(self) -> None:
        from libs.common.deidentification import deidentify_patient

        record: dict[str, Any] = {"relatives": [{"first_name": "Bob"}, {"first_name": "Carol"}]}
        result = deidentify_patient(record)
        for rel in result["relatives"]:
            assert rel["first_name"] == "[REDACTED]"

    def test_is_deidentified_true_after_processing(self) -> None:
        from libs.common.deidentification import deidentify_patient, is_deidentified

        raw = {"first_name": "Alice", "gender": "female"}
        deidentified = deidentify_patient(raw)
        assert is_deidentified(deidentified)

    def test_is_deidentified_false_for_raw_record(self) -> None:
        from libs.common.deidentification import is_deidentified

        assert not is_deidentified({"first_name": "Alice"})

    def test_input_not_mutated(self) -> None:
        from libs.common.deidentification import deidentify_patient

        original: dict[str, Any] = {"first_name": "Alice", "gender": "female"}
        deidentify_patient(original)
        assert original["first_name"] == "Alice"


# ── JWT ────────────────────────────────────────────────────────────────────────


class TestJWT:
    _JWT_ENV = {
        "JWT_SECRET_KEY": "a" * 32,
        "JWT_ALGORITHM": "HS256",
        "JWT_EXPIRE_MINUTES": "60",
    }

    def _patch_settings(self) -> Any:
        return patch.dict(os.environ, self._JWT_ENV)

    def test_create_and_verify(self) -> None:
        with self._patch_settings():
            from libs.common import config as cfg

            cfg.get_settings.cache_clear()
            from services.api.auth.jwt import create_access_token, verify_token

            token, expires_in = create_access_token("alice", "clinician")
            assert isinstance(token, str)
            assert expires_in == 3600
            claims = verify_token(token)
            assert claims.user_id == "alice"
            assert claims.role == "clinician"
            assert len(claims.jti) == 36  # UUID4
            cfg.get_settings.cache_clear()

    def test_expired_token_raises_401(self) -> None:
        import jwt as pyjwt
        from fastapi import HTTPException

        with self._patch_settings():
            from libs.common import config as cfg

            cfg.get_settings.cache_clear()
            # Manually craft an already-expired token
            payload = {
                "sub": "alice",
                "role": "clinician",
                "jti": str(uuid.uuid4()),
                "iat": int(time.time()) - 7200,
                "exp": int(time.time()) - 3600,
            }
            token = pyjwt.encode(payload, "a" * 32, algorithm="HS256")
            from services.api.auth.jwt import verify_token

            with pytest.raises(HTTPException) as exc_info:
                verify_token(token)
            assert exc_info.value.status_code == 401
            assert "expired" in exc_info.value.detail.lower()
            cfg.get_settings.cache_clear()

    def test_tampered_token_raises_401(self) -> None:
        from fastapi import HTTPException

        with self._patch_settings():
            from libs.common import config as cfg

            cfg.get_settings.cache_clear()
            from services.api.auth.jwt import create_access_token, verify_token

            token, _ = create_access_token("alice", "clinician")
            tampered = token[:-4] + "XXXX"
            with pytest.raises(HTTPException) as exc_info:
                verify_token(tampered)
            assert exc_info.value.status_code == 401
            cfg.get_settings.cache_clear()

    def test_wrong_secret_raises_401(self) -> None:
        import jwt as pyjwt
        from fastapi import HTTPException

        with self._patch_settings():
            from libs.common import config as cfg

            cfg.get_settings.cache_clear()
            payload = {
                "sub": "alice",
                "role": "clinician",
                "jti": str(uuid.uuid4()),
                "iat": int(time.time()),
                "exp": int(time.time()) + 3600,
            }
            token = pyjwt.encode(payload, "wrong_secret_key_32_chars_padded_!", algorithm="HS256")
            from services.api.auth.jwt import verify_token

            with pytest.raises(HTTPException) as exc_info:
                verify_token(token)
            assert exc_info.value.status_code == 401
            cfg.get_settings.cache_clear()

    def test_each_token_has_unique_jti(self) -> None:
        with self._patch_settings():
            from libs.common import config as cfg

            cfg.get_settings.cache_clear()
            from services.api.auth.jwt import create_access_token, verify_token

            t1, _ = create_access_token("alice", "clinician")
            t2, _ = create_access_token("alice", "clinician")
            c1 = verify_token(t1)
            c2 = verify_token(t2)
            assert c1.jti != c2.jti
            cfg.get_settings.cache_clear()


# ── RBAC ───────────────────────────────────────────────────────────────────────


class TestRBAC:
    def test_admin_has_all_permissions(self) -> None:
        from services.api.auth.rbac import ROLE_PERMISSIONS, Permission, Role

        admin_perms = ROLE_PERMISSIONS[Role.ADMIN]
        for perm in Permission:
            assert perm in admin_perms

    def test_researcher_has_only_predict_risk(self) -> None:
        from services.api.auth.rbac import ROLE_PERMISSIONS, Permission, Role

        researcher_perms = ROLE_PERMISSIONS[Role.RESEARCHER]
        assert Permission.PREDICT_RISK in researcher_perms
        # Researchers must NOT see family profiles (contains first-degree relative PHI)
        assert Permission.READ_FAMILY_PROFILE not in researcher_perms
        assert Permission.MANAGE_USERS not in researcher_perms
        assert Permission.VIEW_AUDIT_LOG not in researcher_perms

    def test_clinician_can_read_family_profile(self) -> None:
        from services.api.auth.rbac import ROLE_PERMISSIONS, Permission, Role

        assert Permission.READ_FAMILY_PROFILE in ROLE_PERMISSIONS[Role.CLINICIAN]

    def test_clinician_cannot_manage_users(self) -> None:
        from services.api.auth.rbac import ROLE_PERMISSIONS, Permission, Role

        assert Permission.MANAGE_USERS not in ROLE_PERMISSIONS[Role.CLINICIAN]

    def test_service_role_is_read_only_machine_to_machine(self) -> None:
        # The service (M2M) role deliberately does NOT mirror the clinician role:
        # Tier 2 granted clinicians write permissions that automated integrations
        # must not hold. Service is limited to prediction and read access.
        from services.api.auth.rbac import ROLE_PERMISSIONS, Permission, Role

        service_perms = ROLE_PERMISSIONS[Role.SERVICE]
        # Shares read/predict capabilities with clinicians...
        assert Permission.PREDICT_RISK in service_perms
        assert Permission.READ_PATIENT in service_perms
        assert Permission.READ_FAMILY_PROFILE in service_perms
        # ...but never write/mutation permissions.
        assert Permission.WRITE_PATIENT not in service_perms
        assert Permission.WRITE_CLINICAL not in service_perms
        assert Permission.WRITE_ENCOUNTER not in service_perms
        assert service_perms != ROLE_PERMISSIONS[Role.CLINICIAN]

    def test_require_permission_passes_for_allowed_role(self) -> None:
        from services.api.auth.models import UserClaims
        from services.api.auth.rbac import Permission, require_permission

        dep_fn = require_permission(Permission.PREDICT_RISK)
        # Extract the inner check function from the Depends wrapper
        check_fn = dep_fn.dependency

        user = UserClaims(user_id="alice", role="clinician", jti="jti-1")
        result = check_fn(user)
        assert result.user_id == "alice"

    def test_require_permission_raises_403_for_disallowed_role(self) -> None:
        from fastapi import HTTPException

        from services.api.auth.models import UserClaims
        from services.api.auth.rbac import Permission, require_permission

        dep_fn = require_permission(Permission.VIEW_AUDIT_LOG)
        check_fn = dep_fn.dependency

        user = UserClaims(user_id="bob", role="clinician", jti="jti-2")
        with pytest.raises(HTTPException) as exc_info:
            check_fn(user)
        assert exc_info.value.status_code == 403

    def test_require_permission_raises_403_for_unknown_role(self) -> None:
        from fastapi import HTTPException

        from services.api.auth.models import UserClaims
        from services.api.auth.rbac import Permission, require_permission

        dep_fn = require_permission(Permission.PREDICT_RISK)
        check_fn = dep_fn.dependency

        user = UserClaims(user_id="hacker", role="superadmin", jti="jti-3")
        with pytest.raises(HTTPException) as exc_info:
            check_fn(user)
        assert exc_info.value.status_code == 403

    def test_all_roles_covered_in_permission_matrix(self) -> None:
        from services.api.auth.rbac import ROLE_PERMISSIONS, Role

        for role in Role:
            assert role in ROLE_PERMISSIONS, f"Role {role} missing from ROLE_PERMISSIONS"

    def test_permission_matrix_values_are_frozensets(self) -> None:
        from services.api.auth.rbac import ROLE_PERMISSIONS

        for role, perms in ROLE_PERMISSIONS.items():
            assert isinstance(perms, frozenset), f"{role} permissions are not a frozenset"
