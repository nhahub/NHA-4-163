"""SQLAlchemy ORM models for the Healthcare Hereditary Disease Prediction System.

Import order matters — models with FK dependencies must be imported after
their target tables so that ``Base.metadata`` is fully populated before
Alembic ``autogenerate`` or ``create_all`` is called.
"""

from libs.common.models.audit_log import AuditLog
from libs.common.models.base import Base
from libs.common.models.cascade import CascadeScreening, CascadeTask
from libs.common.models.condition import Condition
from libs.common.models.consent import ConsentRecord
from libs.common.models.encounter import Encounter, encounter_participant
from libs.common.models.family_member_history import FamilyMemberHistory
from libs.common.models.genetic_test import GeneticTest, Variant
from libs.common.models.medication_request import MedicationRequest
from libs.common.models.notification import Notification
from libs.common.models.observation import Observation
from libs.common.models.organization import Organization
from libs.common.models.patient import Patient
from libs.common.models.physician import Physician

__all__ = [
    "Base",
    "AuditLog",
    "CascadeScreening",
    "CascadeTask",
    "Condition",
    "ConsentRecord",
    "Encounter",
    "encounter_participant",
    "FamilyMemberHistory",
    "GeneticTest",
    "MedicationRequest",
    "Notification",
    "Observation",
    "Organization",
    "Patient",
    "Physician",
    "Variant",
]
