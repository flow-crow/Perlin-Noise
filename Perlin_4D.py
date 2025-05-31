import numpy as np
from tqdm import tqdm

class Perlin4D:

    def __init__(self, Nx, Ny, Nz, Nt, offset = 1/32):

        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Nt = Nt
        
        # Setup domain
        self.X, self.Y, self.Z = np.meshgrid(np.linspace(0,1,Nx),
                                             np.linspace(0,1,Ny),
                                             np.linspace(0,1,Nz),
                                             indexing = 'xy')

        self.W = np.linspace(0,1,Nt)

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
        self.freq_z = None
        self.freq_t = None
        
        self.octaves = None
        self.persistence = None
        self.lacunarity = None
        self.amplitude = None

    def print_parameters(self):
    # Print perlin noise parameters
    
        print('****** Parameters ******')
        print(f'Resolution (X,Y,Z): ({self.Nx},{self.Ny},{self.Nz})')
        print(f'Number of images: {self.Nt}')
        print(f'Frequency: {self.freq_x},{self.freq_y},{self.freq_z}')
        print(f'Timescale: {self.freq_t}')
        
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

    def grad(self, hash, x, y, z, w):
    # Gradient
       h = hash & 31
       u = np.where(h >> 3 == 1, y, np.where(h >> 3 == 2, w, np.where(h >> 3 == 3, z, x)))
       v = np.where(h >> 3 == 1, z, np.where(h >> 3 == 2, x, np.where(h >> 3 == 3, w, y)))
       c = np.where(h >> 3 == 1, w, np.where(h >> 3 == 2, y, np.where(h >> 3 == 3, x, z)))
       
       u_term = np.where(h & 4 == 0, u, -u)
       v_term = np.where(h & 2 == 0, v, -v)
       c_term = np.where(h & 1 == 0, c, -c)
       
       return u_term + v_term + c_term

    # Fundamental perlin noise calculation
    def gen_Perlin(self, x_in, y_in, z_in, w_in):
    
       X = np.floor(x_in).astype(int) & self.bitwise_val # Same as X = np.floor(x).astype(int) % 256
       Y = np.floor(y_in).astype(int) & self.bitwise_val
       Z = np.floor(z_in).astype(int) & self.bitwise_val
       W = np.floor(w_in).astype(int) & self.bitwise_val
       
       x = x_in - np.floor(x_in)
       y = y_in - np.floor(y_in)
       z = z_in - np.floor(z_in)
       w = w_in - np.floor(w_in)
    
       a = self.fade(x)
       b = self.fade(y)
       c = self.fade(z)
       d = self.fade(w)
    
       A = self.perm_matrix[X & self.bitwise_val] + Y
       B = self.perm_matrix[(X + 1) & self.bitwise_val] + Y
       AA = self.perm_matrix[A & self.bitwise_val] + Z
       AB = self.perm_matrix[(A + 1) & self.bitwise_val] + Z
       BA = self.perm_matrix[B & self.bitwise_val] + Z
       BB = self.perm_matrix[(B + 1) & self.bitwise_val] + Z
    
       AAA = self.perm_matrix[AA & self.bitwise_val] + W
       AAB = self.perm_matrix[(AA + 1) & self.bitwise_val] + W
       ABA = self.perm_matrix[AB & self.bitwise_val] + W
       ABB = self.perm_matrix[(AB + 1) & self.bitwise_val] + W
       BAA = self.perm_matrix[BA & self.bitwise_val] + W
       BAB = self.perm_matrix[(BA + 1) & self.bitwise_val] + W
       BBA = self.perm_matrix[BB & self.bitwise_val] + W
       BBB = self.perm_matrix[(BB + 1) & self.bitwise_val] + W
    
       alerp1 = self.lerp(a, self.grad(self.perm_matrix[AAA & self.bitwise_val], x, y, z, w), self.grad(self.perm_matrix[BAA & self.bitwise_val], x - 1, y, z, w))
       alerp2 = self.lerp(a, self.grad(self.perm_matrix[ABA & self.bitwise_val], x, y - 1, z, w), self.grad(self.perm_matrix[BBA & self.bitwise_val], x - 1, y - 1, z, w))
       alerp3 = self.lerp(a, self.grad(self.perm_matrix[AAB & self.bitwise_val], x, y, z - 1, w), self.grad(self.perm_matrix[BAB & self.bitwise_val], x - 1, y, z - 1, w))
       alerp4 = self.lerp(a, self.grad(self.perm_matrix[ABB & self.bitwise_val], x, y - 1, z - 1, w), self.grad(self.perm_matrix[BBB & self.bitwise_val], x - 1, y - 1, z - 1, w))
       alerp5 = self.lerp(a, self.grad(self.perm_matrix[(AAA + 1) & self.bitwise_val], x, y, z, w - 1), self.grad(self.perm_matrix[(BAA + 1) & self.bitwise_val], x - 1, y, z, w - 1))
       alerp6 = self.lerp(a, self.grad(self.perm_matrix[(ABA + 1) & self.bitwise_val], x, y - 1, z, w - 1), self.grad(self.perm_matrix[(BBA + 1) & self.bitwise_val], x - 1, y - 1, z, w - 1))
       alerp7 = self.lerp(a, self.grad(self.perm_matrix[(AAB + 1) & self.bitwise_val], x, y, z - 1, w - 1), self.grad(self.perm_matrix[(BAB + 1) & self.bitwise_val], x - 1, y, z - 1, w - 1))
       alerp8 = self.lerp(a, self.grad(self.perm_matrix[(ABB + 1) & self.bitwise_val], x, y - 1, z - 1, w - 1), self.grad(self.perm_matrix[(BBB + 1) & self.bitwise_val], x - 1, y - 1, z - 1, w - 1))
       
       blerp1 = self.lerp(b, alerp1, alerp2)
       blerp2 = self.lerp(b, alerp3, alerp4)
       blerp3 = self.lerp(b, alerp5, alerp6)
       blerp4 = self.lerp(b, alerp7, alerp8)
       
       clerp1 = self.lerp(c, blerp1, blerp2)
       clerp2 = self.lerp(c, blerp3, blerp4)
       
       dlerp = self.lerp(d, clerp1, clerp2)
           
       return dlerp

    # Classic perlin noise at a single frequency
    def classic_Perlin(self, freq_x, freq_y, freq_z, freq_t):
    
        self.freq_x = freq_x
        self.freq_y = freq_y
        self.freq_z = freq_z
        self.freq_t = freq_t
        
        self.octaves = None
        self.persistence = None
        self.lacunarity = None
        self.amplitude = None
        
        X_points = (self.X + self.offset) * freq_x
        Y_points = (self.Y + self.offset) * freq_y
        Z_points = (self.Z + self.offset) * freq_z
        W_seq = (self.W + self.offset) * freq_t * (self.Nt - 1)

        final_result = np.zeros((self.Ny,self.Nx,self.Nz,self.Nt))
    
        for i, w in tqdm(enumerate(W_seq)):
     
            W_points = np.ones_like(X_points) * w
     
            result = self.gen_Perlin(X_points, Y_points, Z_points, W_points)
            final_result[:,:,:,i] = result
        
        return final_result
    

    # Fractal perlin noise combines multiple Perlin noise images at increasing frequency and decreasing amplitude.
    def fractal_Perlin(self,
                       ifreq_x, ifreq_y, ifreq_z, freq_t,
                       octaves = 4,
                       initial_amplitude = 1,
                       persistence = 0.5,
                       lacunarity = 2):

        self.freq_x = ifreq_x
        self.freq_y = ifreq_y
        self.freq_z = ifreq_z
        self.freq_t = freq_t
        self.octaves = octaves   # Number of octaves
        self.amplitude = initial_amplitude # Initial amplitude at starting octave
        self.persistence = persistence # Amplitude scaling factor per octave
        self.lacunarity = lacunarity # Frequency scaling factor per octave
        
        X_points = (self.X + self.offset) 
        Y_points = (self.Y + self.offset) 
        Z_points = (self.Z + self.offset) 
        W_seq = (self.W + self.offset) * freq_t * (self.Nt - 1)
        
        final_result = np.zeros((self.Ny,self.Nx,self.Nz,self.Nt))

        for i, w in tqdm(enumerate(W_seq)):

            maxValue = 0
            W_points = np.ones_like(X_points) * w

            result = np.zeros_like(X_points)

            amplitude = initial_amplitude
            freq_x = ifreq_x
            freq_y = ifreq_y
            freq_z = ifreq_z

            for octave in range(octaves):

                result += self.gen_Perlin(X_points * freq_x,
                                          Y_points * freq_y,
                                          Z_points * freq_z,
                                          W_points) * amplitude

                amplitude *= persistence
                freq_x *= lacunarity
                freq_y *= lacunarity
                freq_z *= lacunarity

                maxValue += amplitude
                
            result /= maxValue
            final_result[:,:,:,i] = result

        return final_result


    # Turbulent perlin noise combines multiple Perlin noise images at increasing frequency and decreasing amplitude.
    # to create a "turbulence"-looking field.
    def turb_Perlin(self,
                    ifreq_x, ifreq_y, ifreq_z, freq_t,
                    octaves = 4,
                    initial_amplitude = 1,
                    persistence = 0.5,
                    lacunarity = 2):

        self.freq_x = ifreq_x
        self.freq_y = ifreq_y
        self.freq_z = ifreq_z
        self.freq_t = freq_t
        self.octaves = octaves   # Number of octaves
        self.amplitude = initial_amplitude # Initial amplitude at starting octave
        self.persistence = persistence # Amplitude scaling factor per octave
        self.lacunarity = lacunarity # Frequency scaling factor per octave
        
        X_points = (self.X + self.offset) 
        Y_points = (self.Y + self.offset) 
        Z_points = (self.Z + self.offset) 
        W_seq = (self.W + self.offset) * freq_t * (self.Nt - 1)
        
        final_result = np.zeros((self.Ny,self.Nx,self.Nz,self.Nt))

        for i, w in tqdm(enumerate(W_seq)):
         
            maxValue = 0
            W_points = np.ones_like(X_points) * w
         
            result = np.zeros_like(X_points)
        
            amplitude = initial_amplitude
            freq_x = ifreq_x
            freq_y = ifreq_y
            freq_z = ifreq_z
            
            for octave in range(octaves):
        
                result += np.abs(self.gen_Perlin(X_points * freq_x,
                                                 Y_points * freq_y,
                                                 Z_points * freq_z,
                                                 W_points)) * amplitude
                
                amplitude *= persistence
                freq_x *= lacunarity   
                freq_y *= lacunarity        
                freq_z *= lacunarity        

                maxValue += amplitude
        
            result /= maxValue
            final_result[:,:,:,i] = result
        
        return final_result


    # Ridge perlin noise combines multiple Perlin noise images at increasing frequency and decreasing amplitude
    # to create a ridge-like peaks within the noise field.
    def ridge_Perlin(self,
                     ifreq_x, ifreq_y, ifreq_z, freq_t,
                     octaves = 4,
                     initial_amplitude = 1,
                     persistence = 0.5,
                     lacunarity = 2):
    
        self.freq_x = ifreq_x
        self.freq_y = ifreq_y
        self.freq_z = ifreq_z
        self.freq_t = freq_t
        self.octaves = octaves   # Number of octaves
        self.amplitude = initial_amplitude # Initial amplitude at starting octave
        self.persistence = persistence # Amplitude scaling factor per octave
        self.lacunarity = lacunarity # Frequency scaling factor per octave
        
        X_points = (self.X + self.offset) 
        Y_points = (self.Y + self.offset) 
        Z_points = (self.Z + self.offset) 
        W_seq = (self.W + self.offset) * freq_t * (self.Nt - 1)
        
        final_result = np.zeros((self.Ny,self.Nx,self.Nz,self.Nt))

        for i, w in tqdm(enumerate(W_seq)):

            maxValue = 0
            W_points = np.ones_like(X_points) * w

            result = np.zeros_like(X_points)

            amplitude = initial_amplitude
            freq_x = ifreq_x
            freq_y = ifreq_y
            freq_z = ifreq_z
            
            for octave in range(octaves):

                tmp_result = np.abs(self.gen_Perlin(X_points * freq_x,
                                                 Y_points * freq_y,
                                                 Z_points * freq_z,
                                                 W_points)) * amplitude
   
                tmp_result = 1 - np.abs(tmp_result)
                tmp_result *= tmp_result
                result += tmp_result * amplitude

                amplitude *= persistence
                freq_x *= lacunarity
                freq_y *= lacunarity
                freq_z *= lacunarity

                maxValue += amplitude

            result /= maxValue
            final_result[:,:,:,i] = result

        return final_result
