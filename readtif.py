# import numpy as np
# import cv2

# img = cv2.imread('E:\\lyh\\hr_1\\data\\tif\\TH01-02_P201208269157431_1B_SXZ_3_08_581_124.tif',-1)
# img_8 = (img / 256).astype('uint8')
# cv2.imwrite('E:\\lyh\\hr_1\\data\\tif\\tem.tif', 256-img_8)
from osgeo import gdal

import numpy as np

ds = gdal.Open("name.tif")

channel = np.array(ds.GetRasterBand(1).ReadAsArray())
