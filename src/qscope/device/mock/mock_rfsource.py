from __future__ import annotations

from typing import Sequence

from qscope.device.device import Device


class MockRFSource(Device):  # Protocol compliance checked by role system
    def __init__(self, **config):
        super().__init__(**config)
        self._connected = False
        self._frequency = 0
        self._power = 0
        self._output = False

    def open(self):
        self._connected = True
        return True, "MockRFSource opened"

    def close(self):
        self._connected = False

    def set_freq(self, freq: float) -> None:
        """Set RF frequency in MHz."""
        self._frequency = freq

    def set_power(self, power: float) -> None:
        """Set RF power in dBm."""
        self._power = power

    def set_state(self, state: bool) -> None:
        """Set RF output state."""
        self._output = state

    def set_freq_list(
        self, rf_freqs: Sequence[float], step_time: float = 0.1
    ) -> Sequence[float]:
        """Configure frequency sweep."""
        self._frequency = rf_freqs[0]  # Just set to first frequency
        return rf_freqs

    def reconnect(self) -> None:
        """Reconnect to RF source."""
        pass

    def start_fm_mod(self) -> None:
        """Start frequency modulation mode."""
        pass

    def stop_fm_mod(self) -> None:
        """Stop frequency modulation mode."""
        pass

    def get_freq(self) -> float:
        """Get RF frequency in MHz."""
        return self._frequency

    def get_power(self) -> float:
        """Get RF power in dBm."""
        return self._power

    def get_state(self) -> bool:
        """Get RF output state."""
        return self._output

    def set_f_table(self, freqs: Sequence[float], powers: Sequence[float]) -> None:
        """Set frequency-power table."""
        pass

    def set_trigger(self, trigger: str) -> None:
        """Set RF trigger."""
        pass

    def start_sweep(self) -> None:
        """Start frequency sweep."""
        pass

    def reset_sweep(self) -> None:
        """Reset frequency sweep."""
        pass

    def stop_sweep(self) -> None:
        """Stop frequency sweep."""
        pass
