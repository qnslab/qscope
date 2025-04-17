import matplotlib.pyplot as plt
import pylablib as pll
from pylablib.devices import Andor

from qscope.device import Zyla42

pll.par["devices/only_windows_dlls"] = False
pll.par["devices/dlls/andor_sdk3"] = "/usr/local/lib/libatcore.so"
Andor.get_cameras_number_SDK3()

# cam1 = Andor.AndorSDK3Camera(idx=0)
# cam1.get_all_attribute_values()
# cam1.get_full_info()
# cam1.open()
# cam1.close()

cam2 = Zyla42()
cam2.open()
cam2.set_exposure_time(1)
image = cam2.take_snapshot()

print(cam2.cam.get_all_attribute_values())

cam2.close()
plt.imshow(image, cmap="gray")
plt.colorbar()
plt.show()
