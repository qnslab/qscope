import pytest
from PyQt6.QtWidgets import QApplication, QWidget

from qscope.gui.util.settings import GUISettings
from qscope.gui.widgets import QuComboBox, QuDoubleSpinBox, QuSpinBox
from qscope.gui.widgets.config import WidgetConfig, WidgetType


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication instance for the test session"""
    app = QApplication([])
    yield app
    app.quit()


class MockWidget(QWidget):
    """Mock widget class for testing settings"""

    def __init__(self):
        super().__init__()
        self.spin_box = QuSpinBox(
            WidgetConfig(
                name="test_spin",
                default=10,
                min_value=0,
                max_value=100,
                widget_type=WidgetType.INT,
            )
        )
        self.double_spin = QuDoubleSpinBox(
            WidgetConfig(
                name="test_double",
                default=1.5,
                min_value=0.0,
                max_value=10.0,
                widget_type=WidgetType.FLOAT,
            )
        )
        combo_config = WidgetConfig(
            name="test_combo",
            default="option1",
            choices=["option1", "option2", "option3"],
            widget_type=WidgetType.CHOICE,
        )
        self.combo_box = QuComboBox(combo_config)
        # Initialize combo box items
        self.combo_box.addItems(combo_config.choices)


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create temporary config directory"""
    return tmp_path / ".qscope"


@pytest.fixture
def gui_settings(qapp, temp_config_dir, monkeypatch):
    """Create GUISettings instance with temporary config directory"""
    monkeypatch.setattr(GUISettings, "config_dir", temp_config_dir)
    widget = MockWidget()
    return GUISettings(widget, "TEST_SECTION")


def test_init_creates_config_dir(gui_settings, temp_config_dir):
    """Test that initialization creates config directory"""
    assert temp_config_dir.exists()
    assert temp_config_dir.is_dir()


def test_default_ini_creation(gui_settings, temp_config_dir):
    """Test default.ini file creation"""
    default_ini = temp_config_dir / "default.ini"
    assert default_ini.exists()
    assert default_ini.is_file()


def test_prev_state_ini_creation(gui_settings, temp_config_dir):
    """Test prev_state.ini file creation"""
    prev_state_ini = temp_config_dir / "prev_state.ini"
    assert prev_state_ini.exists()
    assert prev_state_ini.is_file()


def test_save_current_state(gui_settings):
    """Test saving and retrieving widget states"""
    # Set new values
    gui_settings.gui_handle.spin_box.set_value(20)
    gui_settings.gui_handle.double_spin.set_value(2.5)
    gui_settings.gui_handle.combo_box.set_value("option2")

    # Save and reload config
    gui_settings.save_current_state()
    gui_settings.config.read(gui_settings.prev_state_ini)

    # Verify saved values
    assert gui_settings.config["TEST_SECTION"]["test_spin"] == "20"
    assert gui_settings.config["TEST_SECTION"]["test_double"] == "2.5"
    assert gui_settings.config["TEST_SECTION"]["test_combo"] == "option2"


def test_load_prev_state(gui_settings):
    """Test loading previously saved state"""
    gui_settings.gui_handle.spin_box.set_value(30)
    gui_settings.save_current_state()
    gui_settings.gui_handle.spin_box.set_value(10)
    gui_settings.load_prev_state()
    assert gui_settings.gui_handle.spin_box.get_value() == 30


def test_reset_to_defaults(gui_settings):
    """Test resetting widgets to default values"""
    gui_settings.gui_handle.spin_box.set_value(50)
    gui_settings.gui_handle.double_spin.set_value(5.0)
    gui_settings.save_current_state()

    gui_settings.reset_to_defaults()
    gui_settings.load_prev_state()

    assert gui_settings.gui_handle.spin_box.get_value() == 10
    assert gui_settings.gui_handle.double_spin.get_value() == 1.5


def test_make_config_and_add_section(gui_settings):
    """Test config creation and section addition"""
    config = gui_settings.make_config()
    config = gui_settings.add_section(config, "TEST_SECTION")

    assert "TEST_SECTION" in config
    assert "test_spin" in config["TEST_SECTION"]
    assert "test_double" in config["TEST_SECTION"]
    assert "test_combo" in config["TEST_SECTION"]


def test_save_config_custom_path(gui_settings, tmp_path):
    """Test saving config to custom filepath"""
    custom_path = tmp_path / "custom_config.ini"
    config = gui_settings.make_config()
    config = gui_settings.add_section(config, "TEST_SECTION")
    gui_settings.save_config(config, custom_path)

    assert custom_path.exists()
    assert custom_path.is_file()
