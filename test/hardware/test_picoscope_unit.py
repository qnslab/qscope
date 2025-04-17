import pytest

import qscope.util
from qscope.device import Picoscope5000a as Picoscope
from qscope.util import TEST_LOGLEVEL


class TestPicoscope:
    @pytest.fixture()
    def client_log(self):
        qscope.util.start_client_log(log_to_file=True, log_level=TEST_LOGLEVEL)
        yield
        qscope.util.shutdown_client_log()

    @pytest.fixture
    def scope(self):
        # Create a concrete test implementation of Picoscope
        class TestScope(Picoscope):
            VOLTAGE_RANGES = {
                0.01: "PS5000A_10MV",
                0.02: "PS5000A_20MV",
                0.05: "PS5000A_50MV",
                0.1: "PS5000A_100MV",
                0.2: "PS5000A_200MV",
                0.5: "PS5000A_500MV",
                1.0: "PS5000A_1V",
                2.0: "PS5000A_2V",
                5.0: "PS5000A_5V",
                10.0: "PS5000A_10V",
                20.0: "PS5000A_20V",
            }

            def __init__(self, **config_kwargs):
                super().__init__(**config_kwargs)
                self._enabled_channels = []
                self._channel_ranges = {}
                self._buffers = {}

            def open(self):
                self._chandle = 1  # Mock handle
                self._ps = type(
                    "MockPS",
                    (),
                    {
                        "PS5000A_CHANNEL": {
                            "PS5000A_CHANNEL_A": 0,
                            "PS5000A_CHANNEL_B": 1,
                        },
                        "PS5000A_COUPLING": {"PS5000A_DC": 1, "PS5000A_AC": 0},
                        "PS5000A_RANGE": {
                            "PS5000A_10MV": 0,
                            "PS5000A_20MV": 1,
                            "PS5000A_50MV": 2,
                            "PS5000A_100MV": 3,
                            "PS5000A_200MV": 4,
                            "PS5000A_500MV": 5,
                            "PS5000A_1V": 6,
                            "PS5000A_2V": 7,
                            "PS5000A_5V": 8,
                            "PS5000A_10V": 9,
                            "PS5000A_20V": 10,
                        },
                        "PICO_INFO": {
                            "PICO_VARIANT_INFO": 0,
                            "PICO_BATCH_AND_SERIAL": 1,
                            "PICO_HARDWARE_VERSION": 2,
                        },
                        "ps5000aSetChannel": lambda *args: 0,
                        "ps5000aMaximumValue": lambda *args: (
                            setattr(args[1], "value", 32767) or 0
                        ),  # Return 0 for PICO_OK
                        "_CloseUnit": lambda *args: 0,
                        "ps5000aGetUnitInfo": lambda *args: (
                            setattr(args[1], "value", b"PS5000a")
                            if args[4] == 0
                            else setattr(args[1], "value", b"12345")
                            if args[4] == 1
                            else setattr(args[1], "value", b"1")
                        ),
                        "ps5000aGetStreamingLatestValues": lambda *args: (
                            args[1](  # Call streaming callback with simulated overflow
                                handle=1,
                                num_samples=50,
                                start_index=0,
                                overflow=1,  # Set overflow bit for channel A
                                trigger_at=0,
                                triggered=0,
                                auto_stop=0,
                                p_parameter=None,
                            ),
                            0,  # Return PICO_OK
                        )[1],
                    },
                )

        scope = TestScope()
        yield scope
        scope.close()

    @pytest.mark.usefixtures("client_log")
    def test_set_resolution(self, scope):
        """Test resolution setting validation"""
        # Valid resolutions
        for res in [8, 12, 14, 15, 16]:
            scope.set_resolution(res)
            assert scope._resolution == res

        # Invalid resolution
        with pytest.raises(ValueError):
            scope.set_resolution(10)

    @pytest.mark.usefixtures("client_log")
    def test_configure_channels(self, scope):
        """Test channel configuration"""
        scope.open()

        # Test valid configuration
        scope.configure_channels(
            channels=[0, 1], ranges=[2.0, 1.0], coupling=["DC", "AC"]
        )
        assert scope._enabled_channels == [0, 1]
        assert scope._channel_ranges[0] == "PS5000A_2V"
        assert scope._channel_ranges[1] == "PS5000A_1V"

        # Test mismatched lengths
        with pytest.raises(ValueError):
            scope.configure_channels(channels=[0], ranges=[2.0, 1.0])

        # Test invalid voltage range
        with pytest.raises(KeyError):
            scope.configure_channels(
                channels=[0],
                ranges=[3.0],  # Not in VOLTAGE_RANGES
            )
