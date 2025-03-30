# kuwahara-arnold-imager

2 Arnold Imager plugins based on the Kuwahara Filter to achieve a stylized NPR look.

- Kuwahara Filter Imager (classic Kuwahara)
- Polynomial Anisotropic Kuwahara Filter Imager

Designed with OpenCV and the Arnold API, multi-threaded using OpenMP, and built with C++ and CMake. Tested on Windows for Arnold 7.3.

The Anisotropic Kuwahara Imager is based on the paper: [Anisotropic Kuwahara Filtering with Polynomial Weighting Functions](./docs/Anisotropic_Kuwahara_Filtering_Paper.pdf)
