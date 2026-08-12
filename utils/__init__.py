"""
Utils package for the intelligent braking system.
"""

from .preprocessing import (
    ImagePreprocessor,
    CANSignalPreprocessor,
    DataSynchronizer,
    PreprocessingConfig,
    RoadSurface
)

from .physics import (
    VehicleParameters,
    TireParameters,
    VehicleDynamics,
    vehicle_dynamics
)

from .can_bus import (
    CANSignal,
    CANMessage,
    CANSignalDecoder,
    CANBusInterface,
    CANDataLogger,
    get_default_signals
)

from .visualization import (
    TrainingPlotter,
    PredictionVisualizer,
    SystemAnalyzer,
    RealTimeVisualizer
)

from .metrics import (
    ClassificationMetrics,
    RegressionMetrics,
    ControlMetrics,
    SystemMetrics
)

# Version
__version__ = "1.0.0"