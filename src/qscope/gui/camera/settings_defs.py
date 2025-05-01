from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from ..widgets.config import WidgetConfig
from ..widgets.types import WidgetType


class CameraSettings:
    """Comprehensive settings definitions for camera settings"""

    # Camera Control Settings
    exposure_time: WidgetConfig[float] = WidgetConfig(
        name="exposure_time",
        widget_type=WidgetType.FLOAT,
        default=0.030,
        min_value=0.001,
        max_value=100.0,
    )

    binning: WidgetConfig[int] = WidgetConfig(
        name="binning",
        widget_type=WidgetType.CHOICE,
        default="1x1",
        choices=[
            "1x1",
            "2x2",
            "4x4",
            "8x8",
        ],
    )

    image_size_x: WidgetConfig[int] = WidgetConfig(
        name="image_size_x",
        widget_type=WidgetType.INT,
        default=256,
        min_value=1,
        max_value=10000,
    )

    image_size_y: WidgetConfig[int] = WidgetConfig(
        name="image_size_y",
        widget_type=WidgetType.INT,
        default=256,
        min_value=1,
        max_value=10000,
    )

    min_limit: WidgetConfig[int] = WidgetConfig(
        name="min_limit",
        widget_type=WidgetType.INT,
        default=0,
        min_value=0,
        max_value=100000,
    )

    max_limit: WidgetConfig[int] = WidgetConfig(
        name="max_limit",
        widget_type=WidgetType.INT,
        default=10000,
        min_value=0,
        max_value=100000,
    )

    def get_widget_config(self, name: str) -> Optional[WidgetConfig]:
        """Get widget configuration by name"""
        return getattr(self, name, None)

    def validate_all(self) -> Dict[str, bool]:
        """Validate all settings"""
        results = {}
        for field_name, field_value in self.__dataclass_fields__.items():
            if isinstance(field_value.default, WidgetConfig):
                config = field_value.default
                value = getattr(self, field_name, None)
                results[field_name] = config.validate(value)
        return results
