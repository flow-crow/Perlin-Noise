import numpy as np

class Perlin3D:

    def __init__(self, Nx, Ny, Nt, offset = 1/32):
        
        self.Nx = Nx
        self.Ny = Ny
        self.Nt = Nt
        
        # Setup domain
        self.X, self.Y, self.Z = np.meshgrid(np.linspace(0,1,Nx),
                                             np.linspace(0,1,Ny),
                                             np.linspace(0,1,Nt),
                                             indexing = 'xy')
    
        # Setup non-integer offset
        assert (offset % 1 != 0), 'Offset must not be an integer'
        self.offset = offset
    
        # Set permutation matrix
        # Original matrix below. Can also use custom: perm_matrix = np.random.permutation(64)
        self.perm_matrix = np.array([151, 160, 137, 91, 90, 15,
                      131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23,
                      190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33,
                      88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166,
                      77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244,
                      102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196,
                      135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123,
                      5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42,
                      223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
                      129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228,
                      251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107,
                      49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254,
                      138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180])
        self.bitwise_val = self.perm_matrix.shape[0] - 1
    
        self.freq_x = None
        self.freq_y = None
        self.freq_t = None
        
        self.octaves = None
        self.persistence = None
        self.lacunarity = None
        self.amplitude = None

    def print_parameters(self):
    # Print perlin noise parameters
        
        print('****** Parameters ******')
        print(f'Resolution (X,Y): ({self.Nx},{self.Ny})')
        print(f'Number of images: {self.Nz}')
        print(f'Frequency: {self.frequency}')
        print(f'Timescale: {self.timescale}')

        if self.octaves is not None:
            print('* Multiscale parameters *')
            print(f'Octaves: {self.octaves}')#
            print(f'Initial frequency: ({self.freq_x},{self.freq_y})')
            print(f'Initial amplitude: {self.amplitude}')
            print(f'Lacunarity: {self.lacunarity}')
            print(f'Persistance: {self.persistence}')

    def fade(self, t):
    # Fade function
        return t * t * t * (t * (t * 6 - 15) + 10)

    def lerp(self, t, a, b):
    # Linear interpolation
       return a + t * (b - a)

    def grad(self, hash, x, y, z):
    # Gradient
       h = hash & 15
    
       u = np.where(h < 8, x, y)
       v = np.where(h < 4, y, np.where((h == 12) | (h == 14), x, z))
    
       u = np.where(h & 1 == 0, u, -u)
       v = np.where(h & 2 == 0, v, -v)
    
       return u + v

    def gen_Perlin(self, x_in, y_in, z_in):
    # Fundamental perlin noise calculation
    
       # Find unit cube that contains point
       X = np.floor(x_in).astype(int) & self.bitwise_val # Same as X = np.floor(x).astype(int) % 256
       Y = np.floor(y_in).astype(int) & self.bitwise_val
       Z = np.floor(z_in).astype(int) & self.bitwise_val
    
       # Find relative x, y, z of point in cube
       x = x_in - np.floor(x_in)
       y = y_in - np.floor(y_in)
       z = z_in - np.floor(z_in)
    
       # Compute fade curves for each of x, y, z
       u = self.fade(x)
       v = self.fade(y)
       w = self.fade(z)
    
       # Hash coordinates of the 8 cube corners
       A = self.perm_matrix[X & self.bitwise_val] + Y
       AA = self.perm_matrix[A & self.bitwise_val] + Z
       AB = self.perm_matrix[(A + 1) & self.bitwise_val] + Z
       B = self.perm_matrix[(X + 1) & self.bitwise_val] + Y
       BA = self.perm_matrix[(B) & self.bitwise_val] + Z
       BB = self.perm_matrix[(B + 1) & self.bitwise_val] + Z
    
       # Add blended results from 8 corners of cube
       # u interp
       ulerp1 = self.lerp(u, self.grad(self.perm_matrix[(AA) & self.bitwise_val], x, y, z), self.grad(self.perm_matrix[(BA) & self.bitwise_val], x-1, y, z))
       ulerp2 = self.lerp(u, self.grad(self.perm_matrix[(AB) & self.bitwise_val], x, y-1, z), self.grad(self.perm_matrix[(BB) & self.bitwise_val], x-1, y-1, z))
       ulerp3 = self.lerp(u, self.grad(self.perm_matrix[(AA+1) & self.bitwise_val], x, y, z-1), self.grad(self.perm_matrix[(BA+1) & self.bitwise_val], x-1, y, z-1))
       ulerp4 = self.lerp(u, self.grad(self.perm_matrix[(AB+1) & self.bitwise_val], x, y-1, z-1), self.grad(self.perm_matrix[(BB+1) & self.bitwise_val], x-1, y-1, z-1))
       # v interp  
       vlerp1 = self.lerp(v, ulerp1, ulerp2)
       vlerp2 = self.lerp(v, ulerp3, ulerp4)
       # w interp  
       final_result = self.lerp(w, vlerp1, vlerp2)
    
       return final_result

    # Classic perlin noise at a single frequency
    def classic_Perlin(self, freq_x, freq_y, freq_t):        
        
        self.freq_x = freq_x
        self.freq_y = freq_y
        self.freq_t = freq_t
        
        self.octaves = None
        self.persistence = None
        self.lacunarity = None
        self.amplitude = None
    
        X_points = (self.X + self.offset) * freq_x
        Y_points = (self.Y + self.offset) * freq_y
        Z_points = (self.Z + self.offset) * freq_t * (self.Nt - 1)
        
        noise_result = self.gen_Perlin(X_points, Y_points, Z_points)
        
        return noise_result

    # Fractal perlin noise combines multiple Perlin noise images at increasing frequency and decreasing amplitude.
    def fractal_Perlin(self,
                      ifreq_x, ifreq_y, freq_t,
                      octaves = 4,
                      initial_amplitude = 1,
                      persistence = 0.5,
                      lacunarity = 2):
    
        self.freq_x = ifreq_x
        self.freq_y = ifreq_y
        self.freq_t = freq_t
        self.octaves = octaves   # Number of octaves
        self.amplitude = initial_amplitude # Initial amplitude at starting octave
        self.persistence = persistence # Amplitude scaling factor per octave
        self.lacunarity = lacunarity # Frequency scaling factor per octave
        
        X_points = self.X + self.offset
        Y_points = self.Y + self.offset
        Z_points = self.Z + self.offset
     
        maxValue = 0
        amplitude = initial_amplitude
        freq_x = ifreq_x
        freq_y = ifreq_y

        total_result = np.zeros_like(X_points)
    
        for octave in range(octaves):
        
            total_result += self.gen_Perlin(X_points * freq_x,
                                            Y_points * freq_y,
                                            Z_points * freq_t  * (self.Nt - 1)
                                            ) * amplitude  
             
            amplitude *= persistence
            freq_x *= lacunarity
            freq_y *= lacunarity

            maxValue += amplitude
        
        total_result /= maxValue
        
        return total_result 


    # Turbulent perlin noise combines multiple Perlin noise images at increasing frequency and decreasing amplitude.
    # to create a "turbulence"-looking field.
    def turb_Perlin(self,
                    ifreq_x, ifreq_y, freq_t,
                    octaves = 4,
                    initial_amplitude = 1,
                    persistence = 0.5,
                    lacunarity = 2):
        
        self.freq_x = ifreq_x
        self.freq_y = ifreq_y
        self.freq_t = freq_t
        self.octaves = octaves   # Number of octaves
        self.amplitude = initial_amplitude # Initial amplitude at starting octave
        self.persistence = persistence # Amplitude scaling factor per octave
        self.lacunarity = lacunarity # Frequency scaling factor per octave
    
        X_points = self.X + self.offset
        Y_points = self.Y + self.offset
        Z_points = self.Z + self.offset
    
        maxValue = 0
        amplitude = initial_amplitude
        freq_x = ifreq_x
        freq_y = ifreq_y
        
        total_result = np.zeros_like(X_points)
    
        for octave in range(octaves):
        
            total_result += np.abs(self.gen_Perlin(X_points * freq_x,
                                                   Y_points * freq_y,
                                                   Z_points * freq_t * (self.Nt - 1)
                                                   )) * amplitude  
        
            amplitude *= persistence
            freq_x *= lacunarity
            freq_y *= lacunarity

            maxValue += amplitude
        
        total_result /= maxValue
        
        return total_result


    # Ridge perlin noise combines multiple Perlin noise images at increasing frequency and decreasing amplitude
    # to create a ridge-like peaks within the noise field.
    def ridge_Perlin(self,
                    ifreq_x, ifreq_y, freq_t,
                    octaves = 4,
                    initial_amplitude = 1,
                    persistence = 0.5,
                    lacunarity = 2):
    
        self.freq_x = ifreq_x
        self.freq_y = ifreq_y
        self.freq_t = freq_t
        self.octaves = octaves   # Number of octaves
        self.amplitude = initial_amplitude # Initial amplitude at starting octave
        self.persistence = persistence # Amplitude scaling factor per octave
        self.lacunarity = lacunarity # Frequency scaling factor per octave
    
        X_points = self.X + self.offset
        Y_points = self.Y + self.offset
        Z_points = self.Z + self.offset
    
        maxValue = 0
        amplitude = initial_amplitude
        freq_x = ifreq_x
        freq_y = ifreq_y
            
        total_result = np.zeros_like(X_points)
    
        for octave in range(octaves):
        
            tmp_result = self.gen_Perlin(X_points * freq_x,
                                         Y_points * freq_y,
                                         Z_points * freq_t * (self.Nt - 1)
                                         )  
        
            tmp_result = 1 - np.abs(tmp_result)
            tmp_result *= tmp_result
        
            total_result += tmp_result * amplitude
        
            amplitude *= persistence
            freq_x *= lacunarity
            freq_y *= lacunarity
            
            maxValue += amplitude # May not be necessary
    
        total_result /= maxValue
    
        return total_result
