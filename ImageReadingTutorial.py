# Authors: Stefanos Papanikolaou <stephanos.papanikolaou@gmail.com>
# BSD 2-Clause License
# Copyright (c) 2021, PapStatMechMat
# All rights reserved.

import ImageTools as IT
import pylab as plt

img=IT.ReadImage("ScreenShot.png")
img2=IT.CropImage(img,15,0)

fig,ax=plt.subplots(1,2)
ax[0].imshow(img)
ax[1].imshow(img2)

plt.show()
