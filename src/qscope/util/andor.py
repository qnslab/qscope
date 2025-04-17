import os
import platform

import pylablib as pll
from pylablib.devices import Andor

this_folder = os.path.abspath(os.path.realpath(os.path.dirname(__file__)))
lib_folder = os.path.abspath(
    os.path.join(this_folder, *[".." for i in range(3)], "proprietary_artefacts")
)


def get_available_andor_cameras():
    if platform.system() == "Windows":
        pll.par["devices/dlls/andor_sdk3"] = lib_folder
    else:
        pll.par["devices/only_windows_dlls"] = False
        pll.par["devices/dlls/andor_sdk3"] = "/usr/local/lib/libatcore.so"

    n = Andor.get_cameras_number_SDK3()
    print(f"====== Found {n} cameras (Last two are often Simcams)")
    for i in range(n):
        print(f"------ Attempting camera #{i}")
        cam = Andor.AndorSDK3Camera(idx=i)
        info = cam.get_all_attribute_values()
        # info = cam.get_full_info()
        print(info)


def restart_lib():
    if platform.system() == "Windows":
        pll.par["devices/dlls/andor_sdk3"] = lib_folder
    else:
        pll.par["devices/only_windows_dlls"] = False
        pll.par["devices/dlls/andor_sdk3"] = "/usr/local/lib/libatcore.so"

    Andor.AndorSDK3.restart_lib()
