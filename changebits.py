from PIL import Image
import numpy as np
im = Image.open("C:/Users/AIPT-GIGABYTE/Desktop/cloud_test/TH01-02_P201604130000031_1B_SXZ_1_08_617_075.tif")
print(im.getbands())
img_array = np.array(im,dtype='uint8')
img_array.dtype
img2 = Image.fromarray(img_array)
print(img2.getbands())
img2.save("C:/Users/AIPT-GIGABYTE/Desktop/cloud_test/TH01-02_P201604130000031_1B_SXZ_1_08_617_075_visual.tif")



