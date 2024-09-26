'''
This code generates a 3D image of a randomly created
scalar field using a combination of sine and cosine functions.
It defines a cubic grid in 3D space and creates a random function
that combines several sine and cosine terms with varying coefficients. 
The resulting scalar field is visualized by displaying slices of the data
as grayscale images in a 2D plot.
'''
import numpy as np
import random

def random_function(X, Y, Z):
    terms = []
    for _ in range(random.randint(2, 10)):
        coeff_x = random.uniform(-5, 5)
        coeff_y = random.uniform(-5, 5)
        coeff_z = random.uniform(-5, 5)
        func_type = random.choice([np.sin, np.cos])
        term = func_type(coeff_x * X + coeff_y * Y + coeff_z * Z)
        terms.append(term)
    combined = terms[0]
    for term in terms[1:]:
        combined = combined + term if random.choice([True, False]) else combined - term
    return combined

def gen_data(nx, ny, nz):
    x = np.linspace(-np.pi, np.pi, nx)
    y = np.linspace(-np.pi, np.pi, ny)
    z = np.linspace(-np.pi, np.pi, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return random_function(X, Y, Z)