# kuwahara-arnold-imager

Set of two Arnold Imager plugins that use the Kuwahara Filter technique to achieve a stylized NPR look.

## Feature

- **Kuwahara Filter Imager (classic Kuwahara)**  
Implements the classic Kuwahara Filter for a smooth, stylized effect.

- **Polynomial Anisotropic Kuwahara Filter Imager**  
Uses a polynomial anisotropic variation to provide greater artistic control. This imager is based on the research presented in the paper: [Anisotropic Kuwahara Filtering with Polynomial Weighting Functions](./docs/Anisotropic_Kuwahara_Filtering_Paper.pdf).

## Examples

### Maya Demo

![Anisotropic Kuwahara Maya Demo](./examples/recording-demo-maya.gif)  
*Maya Imager Demo, from a simple texture.*

![Anisotropic Kuwahara Dragon Comparaison](./examples/anistropicKuwahara-dragon-comparaisonmesh.jpg)

![Anisotropic Kuwahara Bunny Comparaison](./examples/anistropicKuwahara-bunny-comparaison.jpg)

![Anisotropic Kuwahara Lion Comparaison](./examples/anistropicKuwahara-lion-comparaison.jpg)  
*Lion Photography.*

![Anisotropic Kuwahara Dragon](./examples/anistropicKuwahara-dragon-comparaisonmesh.jpg)

![Anisotropic Kuwahara Dragon](./examples/anistropicKuwahara-dragon.jpg)

![Anisotropic Kuwahara Bunny](./examples/anistropicKuwahara-bunny.jpg)

![Anisotropic Kuwahara Lion](./examples/anistropicKuwahara-lion.jpg)

## Requirements

- C++ 17
- CMake
- Arnold 7.3
- OpenCV
- OpenMP

*Only tested on Windows, compiled with MSVC.*

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

Research paper: [Anisotropic Kuwahara Filtering with Polynomial Weighting Functions](./docs/Anisotropic_Kuwahara_Filtering_Paper.pdf).
