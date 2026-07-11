"""Unit tests for the Mendelian inheritance calculator (Tier 5)."""

from __future__ import annotations

import pytest

from services.api.services.inheritance_service import (
    INHERITANCE_MODELS,
    categorise_relationship,
    compute_relative_risk,
    infer_sex,
)


class TestRelationshipCategorisation:
    def test_parent_codes(self) -> None:
        assert categorise_relationship("MTH") == "parent"
        assert categorise_relationship("fth") == "parent"

    def test_sibling_and_child(self) -> None:
        assert categorise_relationship("SIB") == "sibling"
        assert categorise_relationship("SON") == "child"
        assert categorise_relationship("DAU") == "child"

    def test_cousin_and_spouse(self) -> None:
        assert categorise_relationship("COUSN") == "cousin"
        assert categorise_relationship("WIFE") == "spouse"

    def test_unknown(self) -> None:
        assert categorise_relationship("ZZZ") == "unknown"

    def test_infer_sex_from_code(self) -> None:
        assert infer_sex("MTH", None) == "female"
        assert infer_sex("FTH", None) == "male"

    def test_explicit_sex_overrides(self) -> None:
        assert infer_sex("SIB", "female") == "female"


class TestAutosomalDominant:
    def test_first_degree_carrier_is_half(self) -> None:
        r = compute_relative_risk("autosomal_dominant", "SON", 0.5, penetrance=1.0)
        assert r.carrier_probability == 0.5
        assert r.affected_probability == 0.5

    def test_penetrance_scales_affected(self) -> None:
        r = compute_relative_risk("autosomal_dominant", "SON", 0.5, penetrance=0.8)
        assert r.carrier_probability == 0.5
        assert r.affected_probability == 0.4

    def test_second_degree_is_quarter(self) -> None:
        r = compute_relative_risk("autosomal_dominant", "GRMTH", 0.25, penetrance=1.0)
        assert r.carrier_probability == 0.25

    def test_spouse_is_background_only(self) -> None:
        r = compute_relative_risk(
            "autosomal_dominant", "WIFE", 0.0, penetrance=1.0, carrier_frequency=0.001
        )
        assert r.carrier_probability == 0.001


class TestAutosomalRecessive:
    def test_sibling_canonical_values(self) -> None:
        r = compute_relative_risk("autosomal_recessive", "SIB", 0.5, penetrance=1.0)
        assert r.carrier_probability == 0.5
        assert r.affected_probability == 0.25

    def test_parent_is_obligate_carrier(self) -> None:
        r = compute_relative_risk("autosomal_recessive", "MTH", 0.5)
        assert r.carrier_probability == 1.0

    def test_child_is_obligate_carrier(self) -> None:
        r = compute_relative_risk(
            "autosomal_recessive", "SON", 0.5, carrier_frequency=0.02, penetrance=1.0
        )
        # Carrier ≈ 1 - carrier_freq; affected ≈ carrier_freq.
        assert r.carrier_probability == pytest.approx(0.98)
        assert r.affected_probability == pytest.approx(0.02)


class TestXLinkedRecessive:
    def test_mother_of_affected_male_is_carrier(self) -> None:
        r = compute_relative_risk("x_linked_recessive", "MTH", 0.5, relative_sex="female")
        assert r.carrier_probability == 1.0

    def test_brother_50pct_affected(self) -> None:
        r = compute_relative_risk(
            "x_linked_recessive", "BRO", 0.5, relative_sex="male", penetrance=1.0
        )
        assert r.affected_probability == 0.5

    def test_son_of_affected_male_not_at_risk(self) -> None:
        r = compute_relative_risk("x_linked_recessive", "SON", 0.5, relative_sex="male")
        assert r.affected_probability == 0.0
        assert r.carrier_probability == 0.0

    def test_daughter_of_affected_male_obligate_carrier(self) -> None:
        r = compute_relative_risk("x_linked_recessive", "DAU", 0.5, relative_sex="female")
        assert r.carrier_probability == 1.0


class TestXLinkedDominant:
    def test_daughter_of_affected_male_affected(self) -> None:
        r = compute_relative_risk(
            "x_linked_dominant", "DAU", 0.5, relative_sex="female", penetrance=1.0
        )
        assert r.affected_probability == 1.0

    def test_son_of_affected_male_unaffected(self) -> None:
        r = compute_relative_risk("x_linked_dominant", "SON", 0.5, relative_sex="male")
        assert r.affected_probability == 0.0


class TestMitochondrial:
    def test_child_inherits_maternal_line(self) -> None:
        r = compute_relative_risk("mitochondrial", "SON", 0.5, penetrance=1.0)
        assert r.affected_probability == 1.0

    def test_father_outside_maternal_line(self) -> None:
        r = compute_relative_risk("mitochondrial", "FTH", 0.5, relative_sex="male")
        assert r.affected_probability == 0.0


class TestModelCatalogueAndErrors:
    def test_all_modes_have_defaults(self) -> None:
        for model in INHERITANCE_MODELS.values():
            assert 0.0 <= model.default_penetrance <= 1.0
            assert 0.0 <= model.default_carrier_frequency <= 1.0
            assert model.description

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_relative_risk("not_a_mode", "SON", 0.5)

    def test_degree_inferred_when_missing(self) -> None:
        # No degree supplied → inferred from category (child → 0.5).
        r = compute_relative_risk("autosomal_dominant", "SON", None, penetrance=1.0)
        assert r.carrier_probability == 0.5
