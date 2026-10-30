from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class BoundaryKind(StrEnum):
    ABSORBING = "absorbing"
    AIR = "air"


class BackendKind(StrEnum):
    NUMPY = "numpy"
    NUMBA = "numba"
    CYTHON = "cython"
    OPENCL = "opencl"
    CUPY = "cupy"


class ApodizationKind(StrEnum):
    RECTANGULAR = "rectangular"
    HANN = "hann"
    HAMMING = "hamming"
    TUKEY = "tukey"


class SourcePolarization(StrEnum):
    NORMAL = "normal"
    TANGENTIAL = "tangential"


@dataclass(slots=True)
class Material:
    name: str = "Steel"
    density_kg_m3: float = 7800.0
    longitudinal_speed_m_s: float = 5850.0
    shear_speed_m_s: float = 3220.0

    @property
    def mu_pa(self) -> float:
        return self.density_kg_m3 * self.shear_speed_m_s**2

    @property
    def lambda_pa(self) -> float:
        return self.density_kg_m3 * (
            self.longitudinal_speed_m_s**2 - 2.0 * self.shear_speed_m_s**2
        )


@dataclass(slots=True)
class BoundarySide:
    kind: BoundaryKind = BoundaryKind.ABSORBING
    thickness_mm: float = 2.0

    def __post_init__(self) -> None:
        # Qt stores StrEnum values as strings in QVariant user data on some
        # PySide6 versions.  Normalize at the model boundary so callers may
        # safely provide either the enum member or its serialized value.
        self.kind = BoundaryKind(self.kind)


@dataclass(slots=True)
class Boundaries:
    top: BoundarySide = field(
        default_factory=lambda: BoundarySide(BoundaryKind.AIR, 0.0)
    )
    bottom: BoundarySide = field(
        default_factory=lambda: BoundarySide(BoundaryKind.ABSORBING, 8.0)
    )
    left: BoundarySide = field(default_factory=BoundarySide)
    right: BoundarySide = field(default_factory=BoundarySide)


@dataclass(slots=True)
class Source:
    center_frequency_hz: float = 1.0e6
    amplitude_pa: float = 1.0e6
    cycles: float = 2.0
    # aperture_mm, center_x_mm, and receiver_offset_mm are retained for
    # backwards compatibility with SimNDT Modern 1.1 project files.  For a
    # multi-element array, the physical aperture is derived from the element
    # width and kerf instead of this legacy aperture value.
    aperture_mm: float = 2.0
    center_x_mm: float = 10.0
    receiver_offset_mm: float = 0.0
    element_count: int = 16
    element_width_mm: float = 0.4
    kerf_mm: float = 0.1
    steering_angle_deg: float = 0.0
    focus_depth_mm: float = 10.0
    transmit_apodization: ApodizationKind = ApodizationKind.HANN
    receive_apodization: ApodizationKind = ApodizationKind.HANN
    polarization: SourcePolarization = SourcePolarization.NORMAL
    transmit_active_elements: list[bool] = field(default_factory=list)
    receive_active_elements: list[bool] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.transmit_apodization = ApodizationKind(self.transmit_apodization)
        self.receive_apodization = ApodizationKind(self.receive_apodization)
        self.polarization = SourcePolarization(self.polarization)
        self.transmit_active_elements = [bool(value) for value in self.transmit_active_elements]
        self.receive_active_elements = [bool(value) for value in self.receive_active_elements]

    @property
    def pitch_mm(self) -> float:
        return self.element_width_mm + self.kerf_mm

    @property
    def physical_aperture_mm(self) -> float:
        if self.element_count <= 1:
            return self.aperture_mm
        return self.element_count * self.element_width_mm + (self.element_count - 1) * self.kerf_mm


@dataclass(slots=True)
class CircularDefect:
    enabled: bool = True
    center_x_mm: float = 10.0
    center_y_mm: float = 10.0
    radius_mm: float = 1.0


@dataclass(slots=True)
class Numerics:
    width_mm: float = 20.0
    height_mm: float = 20.0
    duration_us: float = 12.0
    points_per_wavelength: int = 12
    max_frequency_factor: float = 2.0
    cfl: float = 0.45
    backend: BackendKind = BackendKind.NUMPY
    record_snapshots: bool = True
    snapshot_count: int = 120

    def __post_init__(self) -> None:
        self.backend = BackendKind(self.backend)


@dataclass(slots=True)
class ProjectConfig:
    material: Material = field(default_factory=Material)
    boundaries: Boundaries = field(default_factory=Boundaries)
    source: Source = field(default_factory=Source)
    defect: CircularDefect = field(default_factory=CircularDefect)
    numerics: Numerics = field(default_factory=Numerics)

    def validate(self) -> None:
        n = self.numerics
        m = self.material
        s = self.source
        n.backend = BackendKind(n.backend)
        s.transmit_apodization = ApodizationKind(s.transmit_apodization)
        s.receive_apodization = ApodizationKind(s.receive_apodization)
        s.polarization = SourcePolarization(s.polarization)
        for side in (
            self.boundaries.top, self.boundaries.bottom,
            self.boundaries.left, self.boundaries.right,
        ):
            side.kind = BoundaryKind(side.kind)
        if n.width_mm <= 0 or n.height_mm <= 0 or n.duration_us <= 0:
            raise ValueError("Domain width, height and duration must be positive.")
        if n.points_per_wavelength < 6:
            raise ValueError("Use at least 6 points per wavelength; 10-15 is recommended.")
        if not 0.05 <= n.cfl <= 0.70:
            raise ValueError("CFL must be between 0.05 and 0.70 for this solver.")
        if not 2 <= n.snapshot_count <= 1000:
            raise ValueError("Snapshot count must be between 2 and 1000.")
        if m.density_kg_m3 <= 0 or m.longitudinal_speed_m_s <= 0:
            raise ValueError("Density and longitudinal speed must be positive.")
        if m.shear_speed_m_s < 0 or m.lambda_pa < 0:
            raise ValueError("Material speeds produce a negative Lamé parameter.")
        if s.center_frequency_hz <= 0 or s.cycles <= 0 or s.amplitude_pa <= 0:
            raise ValueError("Source frequency, cycles and traction amplitude must be positive.")
        if s.element_count < 1 or s.element_count > 1024:
            raise ValueError("Element count must be between 1 and 1024.")
        for name, mask in (
            ("transmit", s.transmit_active_elements),
            ("receive", s.receive_active_elements),
        ):
            if mask and len(mask) != s.element_count:
                raise ValueError(
                    f"The {name} active-element mask must have one value per element."
                )
            if mask and not any(mask):
                raise ValueError(f"At least one {name} element must be active.")
        if s.element_width_mm <= 0 or s.kerf_mm < 0 or s.aperture_mm <= 0:
            raise ValueError("Element width and aperture must be positive; kerf cannot be negative.")
        if not -89.0 < s.steering_angle_deg < 89.0:
            raise ValueError("Steering angle must be between -89 and 89 degrees.")
        if s.focus_depth_mm < 0:
            raise ValueError("Focus depth cannot be negative; use zero for plane-wave steering.")
        if not 0 <= s.center_x_mm <= n.width_mm:
            raise ValueError("Source center must lie inside the domain.")
        half_aperture = 0.5 * s.physical_aperture_mm
        if s.center_x_mm - half_aperture < 0 or s.center_x_mm + half_aperture > n.width_mm:
            raise ValueError("The physical array aperture exceeds the domain width.")
        if self.defect.enabled:
            d = self.defect
            if d.radius_mm <= 0:
                raise ValueError("Defect radius must be positive.")
            if not (d.radius_mm <= d.center_x_mm <= n.width_mm - d.radius_mm):
                raise ValueError("Defect exceeds the left or right domain edge.")
            if not (d.radius_mm <= d.center_y_mm <= n.height_mm - d.radius_mm):
                raise ValueError("Defect exceeds the top or bottom domain edge.")
        for side in (self.boundaries.top, self.boundaries.bottom,
                     self.boundaries.left, self.boundaries.right):
            if side.thickness_mm < 0:
                raise ValueError("Boundary thickness cannot be negative.")
        if self.boundaries.top.kind is not BoundaryKind.AIR:
            raise ValueError(
                "The verified contact-array model requires a top Air/free-surface boundary."
            )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["numerics"]["backend"] = self.numerics.backend.value
        data["source"]["transmit_apodization"] = self.source.transmit_apodization.value
        data["source"]["receive_apodization"] = self.source.receive_apodization.value
        data["source"]["polarization"] = self.source.polarization.value
        for name in ("top", "bottom", "left", "right"):
            data["boundaries"][name]["kind"] = getattr(self.boundaries, name).kind.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectConfig":
        b = data.get("boundaries", {})
        boundaries = Boundaries(**{
            name: BoundarySide(
                kind=BoundaryKind(b.get(name, {}).get(
                    "kind", "air" if name == "top" else "absorbing"
                )),
                thickness_mm=float(b.get(name, {}).get(
                    "thickness_mm", 0.0 if name == "top" else (8.0 if name == "bottom" else 2.0)
                )),
            )
            for name in ("top", "bottom", "left", "right")
        })
        numerics_data = dict(data.get("numerics", {}))
        numerics_data["backend"] = BackendKind(numerics_data.get("backend", "numpy"))
        source_data = dict(data.get("source", {}))
        # Version 1.1 projects had no array fields.  They remain a single
        # physical element with the saved aperture and centre.
        if "element_count" not in source_data:
            source_data["element_count"] = 1
            source_data["element_width_mm"] = float(source_data.get("aperture_mm", 2.0))
            source_data["kerf_mm"] = 0.0
            source_data["focus_depth_mm"] = 0.0
            source_data["transmit_apodization"] = "hann"
            source_data["receive_apodization"] = "hann"
        return cls(
            material=Material(**data.get("material", {})),
            boundaries=boundaries,
            source=Source(**source_data),
            defect=CircularDefect(**data.get("defect", {})),
            numerics=Numerics(**numerics_data),
        )


MATERIAL_PRESETS: dict[str, Material] = {
    "Steel": Material("Steel", 7800.0, 5850.0, 3220.0),
    "Aluminium": Material("Aluminium", 2700.0, 6320.0, 3130.0),
    "Concrete": Material("Concrete", 2400.0, 4000.0, 2400.0),
    "Water": Material("Water", 1000.0, 1480.0, 0.0),
}
