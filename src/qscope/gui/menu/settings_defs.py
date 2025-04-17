from qscope.gui.widgets.config import WidgetConfig
from qscope.gui.widgets.types import WidgetType


class StatusBarSettings:
    """Comprehensive settings definitions for camera settings"""

    # Camera Control Settings
    project_name: WidgetConfig[str] = WidgetConfig(
        name="project_name",
        widget_type=WidgetType.STRING,
        default="data",
    )
