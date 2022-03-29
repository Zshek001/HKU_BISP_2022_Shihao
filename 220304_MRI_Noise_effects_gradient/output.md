# 3D Image gradient at presence of noise
*Mar 4 , 2022*

In MRI data, a common noise type is the [Rician noise](https://doi.org/10.1002/mrm.1910340618). As explained in [this article](https://doi.org/10.1002/cmr.a.20124), the Rician noise is in essence the result of Gaussian noise on both the real and imaginary channel of complex data, and when calculating the magnitude, the magnitude follows a PDF which is known as the Rician distribution.

In this notebook the effects of four types of noises common in MRI images on the sobel operator are explored:
- Rician
- Gaussian
- K space spike
- Gibbs

#### _load 3D MRI image data_

```
import os
import math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import scipy, scipy.io
from scipy import ndimage, signal

import monai
import h5py
```
```
# data file name
fname = '101915_3T_T2w_SPC1'
fnamemix = '105216_3T_T2w_SPC2'
# Load data
with h5py.File(os.path.join('data', fname+'.h5')) as f:
    x = f['data'][()]
    y = f['target'][()]
# print('Shape of x (input, Low Res) =', x.shape) # (128, 128, 32)
# print('Shape of y (target, High Res) =', y.shape) # (256, 256, 64)
```
```
def visualise_img(image, fig_size=(4, 8), res=500):
    fig = plt.figure(dpi=res)
    grid = ImageGrid(fig, 111, nrows_ncols=fig_size, axes_pad=0.0)
    for i in range(image.shape[-1]):
        #grid[i].imshow(np.rot90(image[i, ...], 1), cmap='gray') # if rot90 here, the translatory transform direction changed
        grid[i].imshow(image[..., i], cmap='gray')
        grid[i].axis('off')
        grid[i].set_xticks([])
        grid[i].set_yticks([])
    plt.show()
```
```
visualise_img(x, (4,8))
```

![](https://github.com/Zshek001/HKU_BISP_2022_Shihao/blob/main/220304_MRI_Noise_effects_gradient/celloutput/1original.png)

## 1. apply noise to data
### Rician
```
noise_rician = monai.transforms.RandRicianNoise(prob=1.0, std=0.8, relative=True, channel_wise=True, sample_std=False)
x_rician = noise_rician(x)
visualise_img(x_rician, (4,8))
```

![](https://github.com/Zshek001/HKU_BISP_2022_Shihao/blob/main/220304_MRI_Noise_effects_gradient/celloutput/2rician.png)

### Gaussian

```
noise_gaussian = monai.transforms.RandGaussianNoise(prob=1.0, mean=0.0, std=800)
x_gaussian = noise_gaussian(x)
visualise_img(x_gaussian, (4,8))
```

![](https://github.com/Zshek001/HKU_BISP_2022_Shihao/blob/main/220304_MRI_Noise_effects_gradient/celloutput/3gaussian.png)
![](https://github.com/Zshek001/HKU_BISP_2022_Shihao/blob/main/220304_MRI_Noise_effects_gradient/celloutput/4kspacespike.png)
![](https://github.com/Zshek001/HKU_BISP_2022_Shihao/blob/main/220304_MRI_Noise_effects_gradient/celloutput/5gibbs.png)
![](https://github.com/Zshek001/HKU_BISP_2022_Shihao/blob/main/220304_MRI_Noise_effects_gradient/celloutput/6sobeloriginal.png)
![](https://github.com/Zshek001/HKU_BISP_2022_Shihao/blob/main/220304_MRI_Noise_effects_gradient/celloutput/7sobelrician.png)
![](https://github.com/Zshek001/HKU_BISP_2022_Shihao/blob/main/220304_MRI_Noise_effects_gradient/celloutput/8sobelgaussian.png)
![](https://github.com/Zshek001/HKU_BISP_2022_Shihao/blob/main/220304_MRI_Noise_effects_gradient/celloutput/9sobelkspacespike.png)
![](https://github.com/Zshek001/HKU_BISP_2022_Shihao/blob/main/220304_MRI_Noise_effects_gradient/celloutput/10sobelgibbs.png)
![](https://github.com/Zshek001/HKU_BISP_2022_Shihao/blob/main/220304_MRI_Noise_effects_gradient/celloutput/11kspacespike_2.png)
![](https://github.com/Zshek001/HKU_BISP_2022_Shihao/blob/main/220304_MRI_Noise_effects_gradient/celloutput/12sobelkspacespike_2.png)
![](https://github.com/Zshek001/HKU_BISP_2022_Shihao/blob/main/220304_MRI_Noise_effects_gradient/celloutput/13fftoriginalpng)
![](https://github.com/Zshek001/HKU_BISP_2022_Shihao/blob/main/220304_MRI_Noise_effects_gradient/celloutput/14fftsobeloriginal.png)
![](https://github.com/Zshek001/HKU_BISP_2022_Shihao/blob/main/220304_MRI_Noise_effects_gradient/celloutput/15comparefftoriginal_sobeloriginal.png)
![](https://github.com/Zshek001/HKU_BISP_2022_Shihao/blob/main/220304_MRI_Noise_effects_gradient/celloutput/16diff_fftsobeloriginal.png)
![](https://github.com/Zshek001/HKU_BISP_2022_Shihao/blob/main/220304_MRI_Noise_effects_gradient/celloutput/17fftKspacespike_2.png)
![](https://github.com/Zshek001/HKU_BISP_2022_Shihao/blob/main/220304_MRI_Noise_effects_gradient/celloutput/18fftsobelKspacespike_2.png)
![](https://github.com/Zshek001/HKU_BISP_2022_Shihao/blob/main/220304_MRI_Noise_effects_gradient/celloutput/19diff_fftsobelKspacespike_2.png)
![](https://github.com/Zshek001/HKU_BISP_2022_Shihao/blob/main/220304_MRI_Noise_effects_gradient/celloutput/20ifft_diff_fftsobelKspacespile_2.png)

