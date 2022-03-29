# MRI image processing

### 22/03/10 *(in progress)*
- 3D optical flow (OF) estimation in MRI data.
  - [Literature Review and Mathematical Basis](https://www.overleaf.com/read/ngfjjdvhzpcd) on mainstream OF methods, namely LK Sparse OF (with corner detector and NMS dependencies), [Farneback Dense OF](https://www.researchgate.net/publication/34757182_Polynomial_Expansion_for_Orientation_and_Motion_Estimation) (the doctorial thesis on polynomial expansion and its application on motion estimation), and [NVIDIA FlowNet2.0](https://github.com/NVIDIA/flownet2-pytorch).
  - re-implementation

### 22/03/04
- 3D Sobel operator performance with noise common in MRI: Rician, Gaussian, K Space Spike, Gibbs. Spectrum analysis on interactions between K Space Spike noise and Sobel operator

### 22/03/02
- 3D Image gradient: (generalized) Sobel operator. Compare official implementation in SciPy.ndimage, OpenCV-python; Implement Sobel myself using SciPy and PyTorch with evaluation.

### 22/01/25 
- Data augmentation for deep learning. Rotation, Translation, Blending, CutOut, CutMix, primarily based on torchvision
- Weighted Patch Sampling for MRI low-/high-res pairs with probing trick. Performance evaluation included.
