import matplotlib.pyplot as plt
from Perlin_4D import Perlin4D
from utils import normalise, save_gif
# %% Setup basic parameters

Nx = 64 
Ny = 64
Nz = 64
N_frames = 20

freq_x = 10
freq_y = freq_x
freq_z = freq_x
freq_t = 0.1

# %% Generate noise fields

Perl = Perlin4D(Nx, Ny, Nz, N_frames)

classic_Perlin = Perl.classic_Perlin(freq_x, freq_y, freq_z, freq_t)
fractal_Perlin = Perl.fractal_Perlin(freq_x, freq_y, freq_z, freq_t)
turb_Perlin    = Perl.turb_Perlin(freq_x, freq_y, freq_z, freq_t)
ridge_Perlin   = Perl.ridge_Perlin(freq_x, freq_y, freq_z, freq_t)

classic_Perlin = normalise(classic_Perlin)
fractal_Perlin = normalise(fractal_Perlin)
turb_Perlin    = normalise(turb_Perlin)
ridge_Perlin   = normalise(ridge_Perlin)
# %% Simple visual animation of a slice through the volume

slice_index = Nz//2
paus_time = 0.1

for i in range(N_frames):

    tmp1 = classic_Perlin[:,:,i]

    plt.figure(0)
    plt.imshow(classic_Perlin[:,:,slice_index,i], cmap = 'viridis', vmin = 0, vmax = 1)
    #plt.imshow(fractal_Perlin[:,:,slice_index,i], cmap = 'viridis', vmin = 0, vmax = 1)
    #plt.imshow(turb_Perlin[:,:,slice_index,i], cmap = 'viridis', vmin = 0, vmax = 1)
    #plt.imshow(ridge_Perlin[:,:,slice_index,i], cmap = 'viridis', vmin = 0, vmax = 1)
    plt.colorbar()
    plt.title(f'Frame: {i+1}/{N_frames}')
    
    plt.pause(paus_time)
    
# %%

save_gif(classic_Perlin, 'classic_perlin_volume_slice', colormap = 'viridis')
save_gif(fractal_Perlin, 'fractal_perlin_volume_slice', colormap = 'viridis')
save_gif(turb_Perlin, 'turbulent_perlin_volume_slice', colormap = 'viridis')
save_gif(ridge_Perlin, 'ridge_perlin_volume_slice', colormap = 'viridis')
