
# Perlin Noise

A fast NumPy implementation of Perlin noise. Generates 3D or 4D procedural noise as an image or volume evolving over time respectively. Includes fractal, turbulent and ridge multiscale forms with tunable parameters.

## Usage

Use example scripts to generate gifs below.
Tunable parameters include:
- ```Nx```,```Ny```,```Nz``` - Noise field size
- ```N_frames``` - Number of noise fields to generate
- ```freq_x```, ```freq_y```, ```freq_z``` - Spatial frequency per dimension
- ```freq_t``` - Temporal frequency
- ```offset``` - Non-integer offset of sampling points
- ```octaves``` - Number of scales to sum 
- ```amplitude``` - Magnitude scaling factor 
- ```lacunarity``` - Frequency scaling factor 

The last three parameters are only relevant for the multiscale forms. Noise fields with parameters ```Nx```, ```Ny```, ```Nz```,```N_frames``` are generated with array dimensions [```Ny```, ```Nx```, ```Nz```,```N_frames```]. Thus, 0th and 1st dimensions correspond to rows (```Ny```) and columns (```Nx```) and the last dimension always corresponds to the temporal dimension (```N_frames```). The temporal dimension is scaled during generation therefore the animation timescale is independent of the number of frames being generated.


## 2D Examples

### Classic Perlin Noise
![classic perlin](gifs/classic_perlin.gif)

### Fractal Perlin Noise
![fractal perlin](gifs/fractal_perlin.gif)

### Turbulent Perlin Noise
![turbulent perlin](gifs/turbulent_perlin.gif)

### Ridge Perlin Noise
![ridge perlin](gifs/ridge_perlin.gif)


## License

This project is licensed under the [MIT License](LICENSE.txt).