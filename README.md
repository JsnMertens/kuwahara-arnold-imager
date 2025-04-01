# kuwahara-arnold-imager

![Static Badge](https://img.shields.io/badge/Arnold-7.3.5-brightgreen?style=flat&logo=autodesk&color=%2342A7CE)
![Static Badge](https://img.shields.io/badge/Arnold-7.4.1-brightgreen?style=flat&logo=autodesk&color=%2342A7CE)



Set of two Arnold Imager plugins that use the [Kuwahara Filter](https://en.wikipedia.org/wiki/Kuwahara_filter) to achieve a stylized [NPR](https://en.wikipedia.org/wiki/Non-photorealistic_rendering) look.

## Feature

- **Kuwahara Filter Imager (classic Kuwahara)**  
Implements the classic Kuwahara Filter for a smooth, stylized effect.

- **Polynomial Anisotropic Kuwahara Filter Imager**  
Uses a polynomial anisotropic variation to provide greater artistic control. This imager is based on the research presented in the paper: [Anisotropic Kuwahara Filtering with Polynomial Weighting Functions](./docs/Anisotropic_Kuwahara_Filtering_Paper.pdf).

## Requirements

- C++ 17
- CMake 3.21+
- Arnold 7.3.5|7.4.0
- OpenCV
- OpenMP

*Only tested on Windows, compiled with MSVC.*

## Examples

![Anisotropic Kuwahara Maya Demo](./examples/recording-demo-maya.gif)  
*Maya Imager Demo, from a simple texture*

![Anisotropic Kuwahara Dragon Comparaison](./examples/anistropicKuwahara-dragon-comparaisonmesh.jpg)

![Anisotropic Kuwahara Bunny Comparaison](./examples/anistropicKuwahara-bunny-comparaison.jpg)

![Anisotropic Kuwahara Lion Comparaison](./examples/anistropicKuwahara-lion-comparaison.jpg)  
*Lion Photography*

![Anisotropic Kuwahara Dragon](./examples/anistropicKuwahara-dragon.jpg)

![Anisotropic Kuwahara Bunny](./examples/anistropicKuwahara-bunny-radius10.jpg)

![Anisotropic Kuwahara Bunny](./examples/anistropicKuwahara-bunny-radius15.jpg)
*Higher radius example (R=15)*

![Anisotropic Kuwahara Lion](./examples/anistropicKuwahara-lion.jpg)
*Lion Photography*

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

Research paper: [Anisotropic Kuwahara Filtering with Polynomial Weighting Functions](./docs/Anisotropic_Kuwahara_Filtering_Paper.pdf).
