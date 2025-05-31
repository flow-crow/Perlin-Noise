import numpy as np
from PIL import Image

def normalise(input_array):
    return (input_array - input_array.min()) / (input_array.max() - input_array.min())
    
def save_gif(scalar, save_name =  None, colormap = 'magma', c_range = 'global', c_min = None, c_max = None):
    
    assert save_name is not None, 'Give a filename.'
    assert c_range in {'global','local','custom'}, f'Invalid input: {c_range}. Must be one either \'global\',\'local\',\'custom\'.'

    if c_min is None and c_max is None:
        S_min = scalar.min()
        S_max = scalar.max()
    if c_range == 'global': # Min and max of full array
        S_min = scalar.min()
        S_max = scalar.max()
    elif c_range == 'custom': # Custom colour limits
        assert c_min is not None and c_max is not None, 'Colour limits must be defined'
        S_min = c_min
        S_max = c_max
    
    print('Saving animation result...')
    T = scalar.shape[2]
    color_vals = plt.get_cmap(colormap)

    frames = []
    for i in range(T):
        S_tmp = scalar[:, :, i]  
        
        if c_range == 'local': # Colour limits of each individual frame
            S_min = S_tmp.min()
            S_max = S_tmp.max()

        S_norm = (S_tmp - S_min) / (S_max - S_min)
        colored_field = color_vals(S_norm)
        colored_field_rgb = (colored_field[:, :, :3] * 255).astype('uint8')
        tmp_image = Image.fromarray(colored_field_rgb)
        frames.append(tmp_image)
        
    frames[0].save(f'{save_name}.gif', save_all=True, append_images=frames[1:], duration=20, loop=0)
    print(f'Saved as: {save_name}.gif')
