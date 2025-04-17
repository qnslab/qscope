"""
Data fitting and analysis for Qscope measurements.

This module provides classes for fitting experimental data to various
models, including:

- Resonance models (Lorentzian, Gaussian)
- Oscillation models (Sine, Damped Sine)
- Decay models (Exponential, Stretched Exponential)
- Linear and polynomial models

Each fitting model provides parameter estimation, uncertainty analysis,
and visualization capabilities.

Examples
--------
Fitting ESR data to a Lorentzian:
```python
from qscope.fitting import Lorentzian
model = Lorentzian()
params, errors = model.fit(x_data, y_data)
print(f"Resonance position: {params[0]} ± {errors[0]}")
```

See Also
--------
qscope.meas : Measurement implementations
"""

# import the different fitting methods
from .decays import (
    ExponentialDecay,
    GuassianDecay,
    StretchedExponentialDecay,
)
from .mpl import MPLFitter
from .oscillations import (
    DampedSine,
    Sine,
)
from .resonance import (
    DifferentialGaussian,
    DifferentialLorentzian,
    Gaussian,
    Linear,
    Lorentzian,
)
