from dataclasses import dataclass, field
from typing import Optional, List
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

@dataclass
class DomainConfig:
    ndim: int = 2
    x_bound: List[float] = field(default_factory=lambda: [0.0, 1.0])
    y_bound: List[float] = field(default_factory=lambda: [0.1, 1.0])
    dx: float = 0.02
    dy: float = 0.02

@dataclass
class ParticlesConfig:
    nparticle_per_dir: int = 4
    x_bounds: List[float] = field(default_factory=lambda: [0.2, 0.3])
    random_x_bounds: bool = True
    y_bounds: List[float] = field(default_factory=lambda: [0.1, 0.5])
    randomness: float = 0.9
    K0: Optional[int] = None
    density:Optional[int] = None
    initial_velocity: Optional[List[float]] = None

@dataclass
class OutputConfig:
    path: str = MISSING

@dataclass
class MpmInputsConfig:
    json_file: str = MISSING
    start_phi: int = 25
    end_phi: int = 35
    increment_phi: float = 5
    n_files_per_phi: int = 6

@dataclass
class Config:
    domain: DomainConfig = field(default_factory=DomainConfig)
    particles: ParticlesConfig = field(default_factory=ParticlesConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    mpm_inputs: MpmInputsConfig = field(default_factory=MpmInputsConfig)

# Hydra configuration
cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)