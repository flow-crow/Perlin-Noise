import matplotlib.pyplot as plt
from Perlin_3D import Perlin3D
from utils import normalise, save_gif
# %% Setup basic parameters

Nx = 512 
Ny = 512 
N_frames = 100 

freq_x = 10
freq_y = freq_x
freq_t = 0.05

# %% Generate noise fields

Perl = Perlin3D(Nx, Ny, N_frames)

classic_Perlin = Perl.classic_Perlin(freq_x, freq_y, freq_t)
fractal_Perlin = Perl.fractal_Perlin(freq_x, freq_y, freq_t)
turb_Perlin    = Perl.turb_Perlin(5, 5, freq_t, octaves = 2)
ridge_Perlin   = Perl.ridge_Perlin(5, 5, freq_t, octaves = 2)

classic_Perlin = normalise(classic_Perlin)
fractal_Perlin = normalise(fractal_Perlin)
turb_Perlin    = normalise(turb_Perlin)
ridge_Perlin   = normalise(ridge_Perlin)

# %% Simple visual animation

paus_time = 0.1

for i in range(N_frames):

    tmp1 = classic_Perlin[:,:,i]

    plt.figure(0)
    plt.imshow(classic_Perlin[:,:,i], cmap = 'viridis', vmin = 0, vmax = 1)
    #plt.imshow(fractal_Perlin[:,:,i], cmap = 'viridis', vmin = 0, vmax = 1)
    #plt.imshow(turb_Perlin[:,:,i], cmap = 'viridis', vmin = 0, vmax = 1)
    #plt.imshow(ridge_Perlin[:,:,i], cmap = 'viridis', vmin = 0, vmax = 1)
    plt.colorbar()
    plt.title(f'Frame: {i+1}/{N_frames}')
    
    plt.pause(paus_time)
    
# %% Save gif

save_gif(classic_Perlin, 'classic_perlin', colormap = 'viridis')
save_gif(fractal_Perlin, 'fractal_perlin', colormap = 'viridis')
save_gif(turb_Perlin, 'turbulent_perlin', colormap = 'viridis')
save_gif(ridge_Perlin, 'ridge_perlin', colormap = 'viridis')
