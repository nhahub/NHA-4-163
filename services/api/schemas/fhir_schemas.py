"""FHIR R4 Pydantic representations for interoperability endpoints.

These models expose our internal ORM records in HL7 FHIR R4 shape so that
external EHR systems (Epic, Cerner, etc.) can consume and submit data using a
standard wire format.  This is a pragmatic subset — we model the resources and
elements the prediction engine actually uses (``Patient``, ``Condition``,
``Observation``, ``Bundle``) rather than the full R4 specification.  Validation
is structural (JSON schema shape) rather than profile-based; full profile
validation would require the heavier ``fhir.resources`` dependency.

FHIR references:
  - Patient:     https://hl7.org/fhir/R4/patient.html
  - Condition:   https://hl7.org/fhir/R4/condition.html
  - Observation: https://hl7.org/fhir/R4/observation.html
  - Bundle:      https://hl7.org/fhir/R4/bundle.html
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ── Primitive / datatype building blocks ──────────────────────────────────────


class FHIRCoding(BaseModel):
    """A single code from a terminology system (FHIR ``Coding``)."""

    system: str | None = None
    code: str | None = None
    display: str | None = None


class FHIRCodeableConcept(BaseModel):
    """A concept described by one or more codings plus optional free text."""

    coding: list[FHIRCoding] = Field(default_factory=list)
    text: str | None = None


class FHIRIdentifier(BaseModel):
    """A business identifier for a resource (FHIR ``Identifier``)."""

    system: str | None = None
    value: str | None = None


class FHIRHumanName(BaseModel):
    """A person's name (FHIR ``HumanName``)."""

    use: str | None = "official"
    family: str | None = None
    given: list[str] = Field(default_factory=list)


class FHIRReference(BaseModel):
    """A reference from one resource to another (FHIR ``Reference``)."""

    reference: str | None = None
    display: str | None = None


class FHIRQuantity(BaseModel):
    """A measured amount (FHIR ``Quantity``)."""

    value: float | None = None
    unit: str | None = None
    system: str | None = None
    code: str | None = None


# ── Resources ─────────────────────────────────────────────────────────────────


class FHIRPatient(BaseModel):
    """FHIR R4 ``Patient`` resource (subset)."""

    resourceType: Literal["Patient"] = "Patient"
    id: str | None = None
    identifier: list[FHIRIdentifier] = Field(default_factory=list)
    active: bool = True
    name: list[FHIRHumanName] = Field(default_factory=list)
    gender: str | None = None
    birthDate: str | None = None
    deceasedBoolean: bool | None = None

    @classmethod
    def from_orm_patient(cls, patient: Any) -> FHIRPatient:
        """Build a FHIR ``Patient`` from a ``libs.common.models.patient.Patient``.

        Args:
            patient: The ORM patient instance.

        Returns:
            Populated :class:`FHIRPatient`.
        """
        identifiers: list[FHIRIdentifier] = []
        if patient.external_id:
            identifiers.append(
                FHIRIdentifier(
                    system=patient.identifier_system or "urn:internal:mrn",
                    value=patient.external_id,
                )
            )

        given: list[str] = []
        if patient.given_name:
            given.append(patient.given_name)
        if patient.middle_name:
            given.append(patient.middle_name)

        gender = patient.gender.value if patient.gender is not None else None

        return cls(
            id=str(patient.id),
            identifier=identifiers,
            active=patient.deleted_at is None,
            name=[FHIRHumanName(family=patient.family_name, given=given)],
            gender=gender,
            birthDate=patient.date_of_birth.isoformat() if patient.date_of_birth else None,
            deceasedBoolean=patient.deceased,
        )


class FHIRCondition(BaseModel):
    """FHIR R4 ``Condition`` resource (subset)."""

    resourceType: Literal["Condition"] = "Condition"
    id: str | None = None
    clinicalStatus: FHIRCodeableConcept | None = None
    verificationStatus: FHIRCodeableConcept | None = None
    code: FHIRCodeableConcept | None = None
    subject: FHIRReference | None = None
    onsetDateTime: str | None = None

    @classmethod
    def from_orm_condition(cls, condition: Any) -> FHIRCondition:
        """Build a FHIR ``Condition`` from an ORM ``Condition``.

        Args:
            condition: The ORM condition instance.

        Returns:
            Populated :class:`FHIRCondition`.
        """
        clinical = FHIRCodeableConcept(
            coding=[
                FHIRCoding(
                    system="http://terminology.hl7.org/CodeSystem/condition-clinical",
                    code=condition.clinical_status.value,
                )
            ]
        )
        verification = None
        if condition.verification_status is not None:
            verification = FHIRCodeableConcept(
                coding=[
                    FHIRCoding(
                        system=("http://terminology.hl7.org/CodeSystem/condition-ver-status"),
                        code=condition.verification_status.value,
                    )
                ]
            )

        return cls(
            id=str(condition.id),
            clinicalStatus=clinical,
            verificationStatus=verification,
            code=FHIRCodeableConcept(
                coding=[
                    FHIRCoding(
                        system=condition.code_system,
                        code=condition.code,
                        display=condition.code_display,
                    )
                ],
                text=condition.code_text or condition.code_display,
            ),
            subject=FHIRReference(reference=f"Patient/{condition.patient_id}"),
            onsetDateTime=(
                condition.onset_datetime.isoformat() if condition.onset_datetime else None
            ),
        )


class FHIRObservation(BaseModel):
    """FHIR R4 ``Observation`` resource (subset)."""

    resourceType: Literal["Observation"] = "Observation"
    id: str | None = None
    status: str = "final"
    category: list[FHIRCodeableConcept] = Field(default_factory=list)
    code: FHIRCodeableConcept | None = None
    subject: FHIRReference | None = None
    effectiveDateTime: str | None = None
    valueQuantity: FHIRQuantity | None = None
    valueString: str | None = None

    @classmethod
    def from_orm_observation(cls, obs: Any) -> FHIRObservation:
        """Build a FHIR ``Observation`` from an ORM ``Observation``.

        Args:
            obs: The ORM observation instance.

        Returns:
            Populated :class:`FHIRObservation`.
        """
        categories: list[FHIRCodeableConcept] = []
        if obs.category:
            categories.append(
                FHIRCodeableConcept(
                    coding=[
                        FHIRCoding(
                            system=(
                                "http://terminology.hl7.org/CodeSystem/" "observation-category"
                            ),
                            code=obs.category,
                        )
                    ]
                )
            )

        value_quantity = None
        if obs.value_quantity is not None:
            value_quantity = FHIRQuantity(
                value=float(obs.value_quantity),
                unit=obs.value_unit,
                system=obs.value_unit_system,
            )

        return cls(
            id=str(obs.id),
            status=obs.status.value,
            category=categories,
            code=FHIRCodeableConcept(
                coding=[
                    FHIRCoding(
                        system=obs.code_system,
                        code=obs.code,
                        display=obs.code_display,
                    )
                ]
            ),
            subject=FHIRReference(reference=f"Patient/{obs.patient_id}"),
            effectiveDateTime=(
                obs.effective_datetime.isoformat() if obs.effective_datetime else None
            ),
            valueQuantity=value_quantity,
            valueString=obs.value_string,
        )


# ── Bundle ────────────────────────────────────────────────────────────────────


class FHIRBundleRequest(BaseModel):
    """The ``request`` element of a transaction bundle entry."""

    method: str = "POST"
    url: str


class FHIRBundleEntry(BaseModel):
    """One entry inside a FHIR ``Bundle``."""

    model_config = ConfigDict(extra="allow")

    fullUrl: str | None = None
    resource: dict[str, Any]
    request: FHIRBundleRequest | None = None


class FHIRBundle(BaseModel):
    """FHIR R4 ``Bundle`` resource (searchset or transaction)."""

    resourceType: Literal["Bundle"] = "Bundle"
    type: str = "searchset"
    total: int | None = None
    entry: list[FHIRBundleEntry] = Field(default_factory=list)

    @classmethod
    def searchset(cls, resources: Sequence[BaseModel]) -> FHIRBundle:
        """Wrap a list of resources in a ``searchset`` bundle.

        Args:
            resources: FHIR resource models to include.

        Returns:
            A :class:`FHIRBundle` of type ``searchset``.
        """
        entries = [FHIRBundleEntry(resource=r.model_dump(exclude_none=True)) for r in resources]
        return cls(type="searchset", total=len(entries), entry=entries)


class FHIRBundleResponseEntry(BaseModel):
    """Result entry returned after processing a transaction bundle."""

    resourceType: str
    id: str
    status: str


class FHIRTransactionResult(BaseModel):
    """Summary returned by ``POST /fhir/Bundle``."""

    resourceType: Literal["Bundle"] = "Bundle"
    type: Literal["transaction-response"] = "transaction-response"
    created: list[FHIRBundleResponseEntry] = Field(default_factory=list)
