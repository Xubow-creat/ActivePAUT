from __future__ import annotations

import pytest

from simndt_modern.models import (
    BackendKind, BoundaryKind, BoundarySide, Numerics, ProjectConfig,
)


def test_project_round_trip() -> None:
    original = ProjectConfig()
    original.numerics.backend = BackendKind.OPENCL
    restored = ProjectConfig.from_dict(original.to_dict())
    assert restored == original


def test_invalid_cfl_is_rejected() -> None:
    config = ProjectConfig()
    config.numerics.cfl = 0.9
    with pytest.raises(ValueError, match="CFL"):
        config.validate()


def test_material_lame_parameters_are_physical() -> None:
    material = ProjectConfig().material
    assert material.mu_pa > 0
    assert material.lambda_pa > 0


def test_string_enum_values_are_normalized() -> None:
    assert Numerics(backend="opencl").backend is BackendKind.OPENCL
    assert BoundarySide(kind="air").kind is BoundaryKind.AIR
