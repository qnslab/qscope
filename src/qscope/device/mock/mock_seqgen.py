from __future__ import annotations

from typing import Any

from qscope.device.device import Device


class MockSeqGen(Device):  # Protocol compliance checked by role system
    def is_connected(self) -> bool:
        return True

    def start(self):
        pass

    def stop(self):
        pass

    def reset(self):
        pass

    def close(self):
        pass

    def open(self):
        return True, "MockSeqGen opened"

    def load_seq(self, seq: Any):
        pass
