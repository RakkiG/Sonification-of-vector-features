import vtk
import numpy as np

def create_vtk_image_data(dims, origin, spacing, func):
    image_data = vtk.vtkImageData()

    image_data.SetDimensions(dims)
    image_data.SetOrigin(origin)
    image_data.SetSpacing(spacing)
    image_data.AllocateScalars(vtk.VTK_DOUBLE, 3)

    vectors = vtk.vtkDoubleArray()
    vectors.SetNumberOfComponents(3)
    vectors.SetNumberOfTuples(dims[0] * dims[1] * dims[2])

    index = 0
    for k in range(dims[2]):
        for j in range(dims[1]):
            for i in range(dims[0]):
                x = i * spacing[0] + origin[0]
                y = j * spacing[1] + origin[1]
                z = k * spacing[2] + origin[2]
                vector = func(x, y, z)
                vectors.SetTuple(index, vector)
                index += 1

    image_data.GetPointData().SetVectors(vectors)

    return image_data


# def vector_field_func(x, y, z):
#     if 0 <= x < 20 and 0 <= y <20:
#         # Source: vectors pointing outward from the center (10, 10, 10)
#         return [(x - 10), (y - 10), (z - 10)]
#     elif 20 <= x < 40 and 0 <= y <20:
#         # Sink: vectors pointing inward towards the center (30, 10, 10)
#         return [8*(30 - x), 9*(10 - y), 5*(10 - z)]
#     elif 40 <= x < 60 and 0 <= y <20:
#         # Saddle point: vectors pointing outward along x and inward along y and z from the center (50, 10, 10)
#         return [(x - 50), -(y - 10), -(z - 10)]
#     elif 60 <= x <= 80 and 0 <= y <20:
#         # Vortex: circular vectors in the xy-plane centered around (70, 10, 10)
#         return [-(y - 10), x - 70, 0]
#
#     elif 0 <= x < 20 and 25 <= y < 40:
#       #（0.3，0.5+0.5i，0.5-0.5i）
#         # Spiral source: outward spiral from the center (10, 30, 10)
#
#         return [0.5 * (x - 10) - 0.5 * (y - 30), 0.5 * (x - 10) + 0.5 * (y - 30), 0.3 * (z - 10)]
#
#     elif 20 <= x < 40 and 25 <= y < 40:
#         # （−0.3，−0.5+0.5i，−0.5-0.5i）
#         # Spiral sink: inward spiral towards the center (30, 30, 10)
#         return [-0.5 * (x - 30) + 0.5 * (y - 30), -0.5 * (x - 30) - 0.5 * (y - 30), -0.3 * (z - 10)]
#
#     elif 40 <= x < 60 and 25 <= y < 40:
#   # （−0.5，0.3+0.3i，0.3-0.3i）
#         # 2:1 Spiral saddle: two directions expanding, one contracting (center: 50, 30, 10)
#         return [0.3 * (x - 50) - 0.3 * (y - 30), 0.3 * (x - 50) + 0.3 * (y - 30), -0.5 * (z - 10)]
#
#     elif 60 <= x <= 80 and 25 <= y < 40:
#         # 1:2 Spiral saddle: one direction expanding, two contracting (center: 70, 30, 10)
#         dx = x - 70
#         dy = y - 30
#         dz = z - 10
#         return [0.5 * dx - 0.3 * dy, 0.3 * dx + 0.5 * dy, -0.8 * dz]
#     else:
#         return [-10, -10, -10]




#----------- For vortex corelines test/show----------
# def vector_field_func(x, y, z):
#     if 0 <= x < 20 and 0 <= y <20:
#         # Source: vectors pointing outward from the center (10, 10, 10)
#         return [(x - 10), (y - 10), (z - 10)]
#     elif 20 <= x < 40 and 0 <= y <20:
#         # Sink: vectors pointing inward towards the center (30, 10, 10)
#         return [8*(30 - x), 9*(10 - y), 5*(10 - z)]
#     elif 40 <= x < 60 and 0 <= y <20:
#         # Saddle point: vectors pointing outward along x and inward along y and z from the center (50, 10, 10)
#         return [(x - 50), -(y - 10), -(z - 10)]
#     elif 60 <= x <= 80 and 0 <= y <20:
#         # Vortex: circular vectors in the xy-plane centered around (70, 10, 10)
#         return [-(y - 10), x - 70, 0]
#
#     elif 0 <= x < 20 and 25 <= y < 40:
#       #（0.3，0.5+0.5i，0.5-0.5i）
#         # Spiral source: outward spiral from the center (10, 30, 10)
#
#         return [0.2 * (x - 10) - 0.2 * (y - 30), 0.2 * (x - 10) + 0.2 * (y - 30), 0.3 * (z - 10)]
#
#     elif 20 <= x < 40 and 25 <= y < 40:
#         # （−0.3，−0.5+0.5i，−0.5-0.5i）
#         # Spiral sink: inward spiral towards the center (30, 30, 10)
#         return [-0.4 * (x - 30) + 0.4 * (y - 30), -0.4 * (x - 30) - 0.4 * (y - 30), -0.3 * (z - 10)]
#
#     elif 40 <= x < 60 and 25 <= y < 40:
#   # （−0.5，0.3+0.3i，0.3-0.3i）
#         # 2:1 Spiral saddle: two directions expanding, one contracting (center: 50, 30, 10)
#         return [0.6 * (x - 50) - 0.6 * (y - 30), 0.6 * (x - 50) + 0.6 * (y - 30), -0.5 * (z - 10)]
#
#     elif 60 <= x <= 80 and 25 <= y < 40:
#         # 1:2 Spiral saddle: one direction expanding, two contracting (center: 70, 30, 10)
#         return [0.8 * (x - 70) - 0.8 * (y - 30), 0.8 * (x - 70) + 0.8 * (y - 30), -0.5 * (z - 10)]
#     else:
#         return [-10, -10, -10]

#----------------------------For critical points test/show------------------------------------
# def vector_field_func(x, y, z):
#     if 0 <= x < 20 and 0 <= y <20:
#         # Source: vectors pointing outward from the center (10, 10, 10)
#         #eigenvalues (5,5,5)
#         return [(x - 10), (y - 10), (z - 10)]
#     elif 20 <= x < 40 and 0 <= y <20:
#         # source: vectors pointing inward towards the center (30, 10, 10)
#         #eigenvalues (10,10,10)
#         return [5 * (x - 30), 5 * (y - 10), 5 * (z - 10)]
#         # return [8*(30 - x), 9*(10 - y), 5*(10 - z)]
#     elif 40 <= x < 60 and 0 <= y <20:
#         # source (50,10,10)
#         # eigenvalues (10,10,10)
#         return [10 * (x-50), 10 * (y-10), 10 * (z-10)]
#
#         # return [(x - 50), -(y - 10), -(z - 10)]
#     elif 60 <= x <= 80 and 0 <= y <20:
#     # source (70,10,10)
#     # eigenvalues (1,5,10)
#         return [1 * (x-70),  5 * (y-10), 10 * (z-10)]
#
#     elif 0 <= x < 20 and 25 <= y < 40:
#         # sink (10,30,10)
#         # eigenvalues (-1,-1,-1)
#
#         return [-1 * (10-x),  -1 * (30-y), -1 * (10-z)]
#
#     elif 20 <= x < 40 and 25 <= y < 40:
#         # sink (30,30,10)
#         # eigenvalues (-5,-5,-5)
#         return [-5 * (30-x) + (-5) * (30-y), -5 * ( 30-x) - 5 * ( 30-y), -5 * (10-z)]
#
#     elif 40 <= x < 60 and 25 <= y < 40:
#   #sink:(50,30,10)
#   #eigenvalues: (-10,-10,-10)
#         return [-10 * ( 50-x) - 10 * (30-y), -10 * (50-x) -10 * (30-y), -10 * (10-z)]
#
#     elif 60 <= x <= 80 and 25 <= y < 40:
#     # sink:(70,30,10)
#     # eigenvalues: (-1,-5,-10)
#         return [- (70-x) - 5 * (30-y), - (70-x) -5* (30-y), -0.5 * (10-z)]
#
#     elif 0 <= x < 20 and 45 <= y < 60:
#         # saddle (10,50,10)
#         # eigenvalues (5,5,-5)
#         return [5 * (x-10), 5 * (y-50), -5 * (z-10)]
#
#     elif 20 <= x < 40 and 45 <= y < 60:
#         # saddle (30,50,10)
#         # eigenvalues (5,-5,-5)
#         return [5 * (x - 30) - 5 * (y - 50), -5 * (x - 30) - 5 * (y - 50), -5 * (z - 10)]
#
#     elif 40 <= x < 60 and 45 <= y < 60:
#         # saddle:(50,50,10)
#         # eigenvalues: (-5,5,10)
#         return [-5 * (x - 50) + 5 * (y - 50), 5 * (x - 50) - 5 * (y - 50), 10 * (z - 10)]
#
#     elif 60 <= x <= 80 and 45 <= y < 60:
#         # saddle:(70,50,10)
#         # eigenvalues: (-10,10,10)
#         return [-10 * (x - 70) + 10 * (y - 50), 10 * (x - 70) - 10 * (y - 50), 10 * (z - 10)]
#
#     elif 0 <= x <= 20 and 65 <= y < 80:
#         # saddle:(10,70,10)
#         # eigenvalues: (-10,-10,10)
#         return [-10 * (x - 10), -10 * (y - 70), 10 * (z - 10)]
#
#     elif 20 <= x <= 40 and 65 <= y < 80:
#         # saddle:(30,70,10)
#         # eigenvalues: (-5,5,-10)
#         return [-5 * (x - 20) + 5 * (y - 70), 5 * (x - 20) - 5 * (y - 70), -10 * (z - 10)]
#
#
#     else:
#         return [-10, -10, -10]



# def vector_field_func(x, y, z):
#     if 0 <= x < 20 and 0 <= y < 20:
#         # Source: vectors pointing outward from the center (10, 10, 10)
#         # Eigenvalues: (5, 5, 5)
#         return [5 * (x - 10), 5 * (y - 10), 5 * (z - 10)]
#
#     elif 20 <= x < 40 and 0 <= y < 20:
#         # Source: vectors pointing inward towards the center (30, 10, 10)
#         # Eigenvalues: (10, 10, 10)
#         return [5 * (x - 30), 5 * (y - 10), 5 * (z - 10)]
#
#     elif 40 <= x < 60 and 0 <= y < 20:
#         # Source: (50, 10, 10)
#         # Eigenvalues: (10, 10, 10)
#         return [10 * (x - 50), 10 * (y - 10), 10 * (z - 10)]
#
#     elif 60 <= x <= 80 and 0 <= y < 20:
#         # Source: (70, 10, 10)
#         # Eigenvalues: (1, 5, 10)
#         return [1 * (x - 70), 5 * (y - 10), 10 * (z - 10)]
#
#     elif 0 <= x < 20 and 25 <= y < 40:
#         # Sink: (10, 30, 10)
#         # Eigenvalues: (-1, -1, -1)
#         return [-1 * (x - 10), -1 * (y - 30), -1 * (z - 10)]
#
#     elif 20 <= x < 40 and 25 <= y < 40:
#         # Sink: (30, 30, 10)
#         # Eigenvalues: (-5, -5, -5)
#         return [-5 * (x - 30), -5 * (y - 30), -5 * (z - 10)]
#
#     elif 40 <= x < 60 and 25 <= y < 40:
#         # Sink: (50, 30, 10)
#         # Eigenvalues: (-10, -10, -10)
#         return [-10 * (x - 50), -10 * (y - 30), -10 * (z - 10)]
#
#     elif 60 <= x <= 80 and 25 <= y < 40:
#         # Sink: (70, 30, 10)
#         # Eigenvalues: (-1, -5, -10)
#         return [-1 * (x - 70), -5 * (y - 30), -10 * (z - 10)]
#
#     elif 0 <= x < 20 and 45 <= y < 60:
#         # Saddle: (10, 50, 10)
#         # Eigenvalues: (5, 5, -5)
#         return [5 * (x - 10), 5 * (y - 50), -5 * (z - 10)]
#
#     elif 20 <= x < 40 and 45 <= y < 60:
#         # Saddle: (30, 50, 10)
#         # Eigenvalues: (5, -5, -5)
#         return [5 * (x - 30), -5 * (y - 50), -5 * (z - 10)]
#
#     elif 40 <= x < 60 and 45 <= y < 60:
#         # Saddle: (50, 50, 10)
#         # Eigenvalues: (-5, 5, 10)
#         return [-5 * (x - 50), 5 * (y - 50), 10 * (z - 10)]
#
#     elif 60 <= x <= 80 and 45 <= y < 60:
#         # Saddle: (70, 50, 10)
#         # Eigenvalues: (-10, 10, 10)
#         return [-10 * (x - 70), 10 * (y - 50), 10 * (z - 10)]
#
#     elif 0 <= x <= 20 and 65 <= y < 80:
#         # Saddle: (10, 70, 10)
#         # Eigenvalues: (-10, -10, 10)
#         return [-10 * (x - 10), -10 * (y - 70), 10 * (z - 10)]
#
#     elif 20 <= x <= 40 and 65 <= y < 80:
#         # Saddle: (30, 70, 10)
#         # Eigenvalues: (-5, 5, -10)
#         return [-5 * (x - 20), 5 * (y - 70), -10 * (z - 10)]
#
#     else:
#         return [-10, -10, -10]




import math

# def vector_field_func(x, y, z):
#     if 0 <= x < 20 and 0 <= y < 20:
#         # Spiral source: vectors spiraling outward from the center (10, 10, 10)
#         # Eigenvalues: (5 + i, 5 - i, 5)
#         dx = 5 * (x - 10) - (y - 10)  # Add rotation in xy-plane
#         dy = 5 * (y - 10) + (x - 10)
#         dz = 5 * (z - 10)
#         return [dx, dy, dz]
#
#     elif 20 <= x < 40 and 0 <= y < 20:
#         # Spiral sink: vectors spiraling inward towards the center (30, 10, 10)
#         # Eigenvalues: (-10 + i, -10 - i, -10)
#         dx = -10 * (x - 30) - (y - 10)  # Add rotation in xy-plane
#         dy = -10 * (y - 10) + (x - 30)
#         dz = -10 * (z - 10)
#         return [dx, dy, dz]
#
#     elif 40 <= x < 60 and 0 <= y < 20:
#         # Radial source: vectors pointing directly outward (50, 10, 10)
#         # Eigenvalues: (10, 10, 10)
#         return [10 * (x - 50), 10 * (y - 10), 10 * (z - 10)]
#
#     elif 60 <= x <= 80 and 0 <= y < 20:
#         # Spiral saddle: combination of outward spiral and inward dynamics in z
#         # Eigenvalues: (1 + i, 1 - i, -10)
#         dx = (x - 70) - (y - 10)  # Add rotation in xy-plane
#         dy = (y - 10) + (x - 70)
#         dz = -10 * (z - 10)
#         return [dx, dy, dz]
#
#     elif 0 <= x < 20 and 25 <= y < 40:
#         # Radial sink: vectors pointing directly inward (10, 30, 10)
#         # Eigenvalues: (-1, -1, -1)
#         return [-1 * (x - 10), -5 * (y - 30), -10 * (z - 10)]
#
#     elif 20 <= x < 40 and 25 <= y < 40:
#         # Spiral sink: vectors spiraling inward (30, 30, 10)
#         # Eigenvalues: (-5 + i, -5 - i, -5)
#         dx = -5 * (x - 30) - (y - 30)
#         dy = -5 * (y - 30) + (x - 30)
#         dz = -5 * (z - 10)
#         return [dx, dy, dz]
#
#     elif 40 <= x < 60 and 25 <= y < 40:
#         # Radial sink: vectors pointing directly inward (50, 30, 10)
#         # Eigenvalues: (-10, -10, -10)
#         return [-10 * (x - 50), -5 * (y - 30), -1 * (z - 10)]
#
#     elif 60 <= x <= 80 and 25 <= y < 40:
#         # Spiral saddle: outward spiral in xy-plane and inward in z
#         # Eigenvalues: (-1 + i, -1 - i, -10)
#         dx = -1 * (x - 70) - (y - 30)
#         dy = -1 * (y - 30) + (x - 70)
#         dz = -10 * (z - 10)
#         return [dx, dy, dz]
#
#     elif 0 <= x < 20 and 45 <= y < 60:
#         # Saddle: (10, 50, 10)
#         # Eigenvalues: (5, 5, -5)
#         return [5 * (x - 10), 5 * (y - 50), -5 * (z - 10)]
#
#     elif 20 <= x < 40 and 45 <= y < 60:
#         # Saddle: (30, 50, 10)
#         # Eigenvalues: (5, -5, -5)
#         return [5 * (x - 30), -5 * (y - 50), -5 * (z - 10)]
#
#     elif 40 <= x < 60 and 45 <= y < 60:
#         # Spiral saddle: outward in xy-plane, inward in z (50, 50, 10)
#         # Eigenvalues: (-5 + i, -5 - i, 10)
#         dx = -5 * (x - 50) - (y - 50)
#         dy = -5 * (y - 50) + (x - 50)
#         dz = 10 * (z - 10)
#         return [dx, dy, dz]
#
#     elif 60 <= x <= 80 and 45 <= y < 60:
#         # Spiral source: outward spiral (70, 50, 10)
#         # Eigenvalues: (10 + i, 10 - i, 10)
#         dx = 10 * (x - 70) - (y - 50)
#         dy = 10 * (y - 50) + (x - 70)
#         dz = 10 * (z - 10)
#         return [dx, dy, dz]
#
#     elif 0 <= x <= 20 and 65 <= y < 80:
#         # Spiral sink: inward spiral (10, 70, 10)
#         # Eigenvalues: (-10 + i, -10 - i, 10)
#         dx = -10 * (x - 10) - (y - 70)
#         dy = -10 * (y - 70) + (x - 10)
#         dz = 10 * (z - 10)
#         return [dx, dy, dz]
#
#     elif 20 <= x <= 40 and 65 <= y < 80:
#         # Saddle: inward spiral in xy-plane and outward in z (30, 70, 10)
#         # Eigenvalues: (-5 + i, -5 - i, -10)
#         dx = -5 * (x - 30) - (y - 70)
#         dy = -5 * (y - 70) + (x - 30)
#         dz = -10 * (z - 10)
#         return [dx, dy, dz]
#
#     else:
#         return [-10, -10, -10]








# def vector_field_func(x, y, z):
#     # Source: vectors pointing outward from the center (10, 10, 10)
#     return [3*(x - 10), -2*(y - 10), (z - 10)]

def vector_field_func(x, y, z):
    # Source: vectors pointing outward from the center (10, 10, 10)
    return [(x-30), (y-30), z-30]

# def vector_field_func(x, y, z):
#     # 区域 1: 0 <= x < 20, 0 <= y < 20
#     if 0 <= x < 20 and 0 <= y < 20:
#         # Vortex: Coreline center at (10, 10, 10), imaginary part |λ_imag| = 0.1
#         return [-(y - 10), x - 10, 0.1 * (z - 10)]
#
#     # 区域 2: 20 <= x < 40, 0 <= y < 20
#     elif 20 <= x < 40 and 0 <= y < 20:
#         # Vortex: Coreline center at (30, 10, 10), imaginary part |λ_imag| = 0.2
#         return [-(y - 10), x - 30, 0.2 * (z - 10)]
#
#     # 区域 3: 40 <= x < 60, 0 <= y < 20
#     elif 40 <= x < 60 and 0 <= y < 20:
#         # Vortex: Coreline center at (50, 10, 10), imaginary part |λ_imag| = 0.3
#         return [-(y - 10), x - 50, 0.3 * (z - 10)]
#
#     # 区域 4: 60 <= x <= 80, 0 <= y < 20
#     elif 60 <= x <= 80 and 0 <= y < 20:
#         # Vortex: Coreline center at (70, 10, 10), imaginary part |λ_imag| = 0.4
#         return [-(y - 10), x - 70, 0.4 * (z - 10)]
#
#     # 区域 5: 0 <= x < 20, 25 <= y < 40
#     elif 0 <= x < 20 and 25 <= y < 40:
#         # Vortex: Coreline center at (10, 30, 10), imaginary part |λ_imag| = 0.5
#         return [-(y - 30), x - 10, 0.5 * (z - 10)]
#
#     # 区域 6: 20 <= x < 40, 25 <= y < 40
#     elif 20 <= x < 40 and 25 <= y < 40:
#         # Vortex: Coreline center at (30, 30, 10), imaginary part |λ_imag| = 0.6
#         return [-(y - 30), x - 30, 0.6 * (z - 10)]
#
#     # 区域 7: 40 <= x < 60, 25 <= y < 40
#     elif 40 <= x < 60 and 25 <= y < 40:
#         # Vortex: Coreline center at (50, 30, 10), imaginary part |λ_imag| = 0.7
#         return [-(y - 30), x - 50, 0.7 * (z - 10)]
#
#     # 区域 8: 60 <= x <= 80, 25 <= y < 40
#     elif 60 <= x <= 80 and 25 <= y < 40:
#         # Vortex: Coreline center at (70, 30, 10), imaginary part |λ_imag| = 0.8
#         return [-(y - 30), x - 70, 0.8 * (z - 10)]
#
#     # 区域 9: 0 <= x < 20, 45 <= y < 60
#     elif 0 <= x < 20 and 45 <= y < 60:
#         # Vortex: Coreline center at (10, 50, 10), imaginary part |λ_imag| = 0.9
#         return [-(y - 50), x - 10, 0.9 * (z - 10)]
#
#     # 区域 10: 20 <= x < 40, 45 <= y < 60
#     elif 20 <= x < 40 and 45 <= y < 60:
#         # Vortex: Coreline center at (30, 50, 10), imaginary part |λ_imag| = 1.0
#         return [-(y - 50), x - 30, 1.0 * (z - 10)]
#
#     # Default: Outside all regions
#     else:
#         return [-10, -10, -10]




# def vector_field_func(x, y, z):
#     if 0 <= x < 10 and 0 <= y < 10:
#         # Source: vectors pointing outward from the center (5, 5, 10)
#         # Eigenvalues: (5, 5, 5)
#         return [5 * (x - 5), 5 * (y - 5), 5 * (z - 10)]
#
#     elif 10 <= x < 20 and 0 <= y < 10:
#         # Source: vectors pointing inward towards the center (15, 5, 10)
#         # Eigenvalues: (10, 10, 10)
#         return [-10 * (x - 15), -10 * (y - 5), -10 * (z - 10)]
#
#     elif 20 <= x < 30 and 0 <= y < 10:
#         # Source: (25, 5, 10)
#         # Eigenvalues: (10, 10, 10)
#         return [10 * (x - 25), 10 * (y - 5), 10 * (z - 10)]
#
#     elif 30 <= x < 40 and 0 <= y < 10:
#         # Source: (35, 5, 10)
#         # Eigenvalues: (1, 5, 10)
#         return [1 * (x - 35), 5 * (y - 5), 10 * (z - 10)]
#
#     elif 0 <= x < 10 and 10 <= y < 20:
#         # Sink: (5, 15, 10)
#         # Eigenvalues: (-1, -1, -1)
#         return [-1 * (x - 5), -1 * (y - 15), -1 * (z - 10)]
#
#     elif 10 <= x < 20 and 10 <= y < 20:
#         # Sink: (15, 15, 10)
#         # Eigenvalues: (-5, -5, -5)
#         return [-5 * (x - 15), -5 * (y - 15), -5 * (z - 10)]
#
#     elif 20 <= x < 30 and 10 <= y < 20:
#         # Sink: (25, 15, 10)
#         # Eigenvalues: (-10, -10, -10)
#         return [-10 * (x - 25), -10 * (y - 15), -10 * (z - 10)]
#
#     elif 30 <= x < 40 and 10 <= y < 20:
#         # Sink: (35, 15, 10)
#         # Eigenvalues: (-1, -5, -10)
#         return [-1 * (x - 35), -5 * (y - 15), -10 * (z - 10)]
#
#     elif 0 <= x < 10 and 20 <= y < 30:
#         # Saddle: (5, 25, 10)
#         # Eigenvalues: (5, 5, -5)
#         return [5 * (x - 5), 5 * (y - 25), -5 * (z - 10)]
#
#     elif 10 <= x < 20 and 20 <= y < 30:
#         # Saddle: (15, 25, 10)
#         # Eigenvalues: (5, -5, -5)
#         return [5 * (x - 15), -5 * (y - 25), -5 * (z - 10)]
#
#     elif 20 <= x < 30 and 20 <= y < 30:
#         # Saddle: (25, 25, 10)
#         # Eigenvalues: (-5, 5, 10)
#         return [-5 * (x - 25), 5 * (y - 25), 10 * (z - 10)]
#
#     elif 30 <= x < 40 and 20 <= y < 30:
#         # Saddle: (35, 25, 10)
#         # Eigenvalues: (-10, 10, 10)
#         return [-10 * (x - 35), 10 * (y - 25), 10 * (z - 10)]
#
#     elif 0 <= x < 10 and 30 <= y < 40:
#         # Saddle: (5, 35, 10)
#         # Eigenvalues: (-10, -10, 10)
#         return [-10 * (x - 5), -10 * (y - 35), 10 * (z - 10)]
#
#     elif 10 <= x < 20 and 30 <= y < 40:
#         # Saddle: (15, 35, 10)
#         # Eigenvalues: (-5, 5, -10)
#         return [-5 * (x - 15), 5 * (y - 35), -10 * (z - 10)]
#
#     else:
#         return [-10, -10, -10]




# def vector_field_func(x, y, z):
#     if 0 <= x < 10 and 0 <= y < 10:
#         # Source: vectors pointing outward from the center (5, 5, 10)
#         # Eigenvalues: (5, 5, 5)
#         return [5 * (x - 5), 5 * (y - 5), 5 * (z - 10)]
#
#     elif 10 <= x < 20 and 0 <= y < 10:
#         # Source: vectors pointing inward towards the center (15, 5, 10)
#         # Eigenvalues: (10, 10, 10)
#         return [-10 * (x - 15), -10 * (y - 5), -10 * (z - 10)]
#
#     elif 20 <= x < 30 and 0 <= y < 10:
#         # Source: (25, 5, 10)
#         # Eigenvalues: (10, 10, 10)
#         return [10 * (x - 25), 10 * (y - 5), 10 * (z - 10)]
#
#     elif 30 <= x < 40 and 0 <= y < 10:
#         # Source: (35, 5, 10)
#         # Eigenvalues: (1, 5, 10)
#         return [1 * (x - 35), 5 * (y - 5), 10 * (z - 10)]
#
#     elif 0 <= x < 10 and 10 <= y < 20:
#         # Sink: (5, 15, 10)
#         # Eigenvalues: (-1, -1, -1)
#         return [-1 * (x - 5), -1 * (y - 15), -1 * (z - 10)]
#
#     elif 10 <= x < 20 and 10 <= y < 20:
#         # Sink: (15, 15, 10)
#         # Eigenvalues: (-5, -5, -5)
#         return [-5 * (x - 15), -5 * (y - 15), -5 * (z - 10)]
#
#     elif 20 <= x < 30 and 10 <= y < 20:
#         # Sink: (25, 15, 10)
#         # Eigenvalues: (-10, -10, -10)
#         return [-10 * (x - 25), -10 * (y - 15), -10 * (z - 10)]
#
#     elif 30 <= x < 40 and 10 <= y < 20:
#         # Sink: (35, 15, 10)
#         # Eigenvalues: (-1, -5, -10)
#         return [-1 * (x - 35), -5 * (y - 15), -10 * (z - 10)]
#
#     elif 0 <= x < 10 and 20 <= y < 30:
#         # Saddle: (5, 25, 10)
#         # Eigenvalues: (5, 5, -5)
#         return [5 * (x - 5), 5 * (y - 25), -5 * (z - 10)]
#
#     elif 10 <= x < 20 and 20 <= y < 30:
#         # Saddle: (15, 25, 10)
#         # Eigenvalues: (5, -5, -5)
#         return [5 * (x - 15), -5 * (y - 25), -5 * (z - 10)]
#
#     elif 20 <= x < 30 and 20 <= y < 30:
#         # Saddle: (25, 25, 10)
#         # Eigenvalues: (-5, 5, 10)
#         return [-5 * (x - 25), 5 * (y - 25), 10 * (z - 10)]
#
#     elif 30 <= x < 40 and 20 <= y < 30:
#         # Saddle: (35, 25, 10)
#         # Eigenvalues: (-10, 10, 10)
#         return [-10 * (x - 35), 10 * (y - 25), 10 * (z - 10)]
#
#     elif 0 <= x < 10 and 30 <= y < 40:
#         # Saddle: (5, 35, 10)
#         # Eigenvalues: (-10, -10, 10)
#         return [-10 * (x - 5), -10 * (y - 35), 10 * (z - 10)]
#
#     elif 10 <= x < 20 and 30 <= y < 40:
#         # Saddle: (15, 35, 10)
#         # Eigenvalues: (-5, 5, -10)
#         return [-5 * (x - 15), 5 * (y - 35), -10 * (z - 10)]
#
#     else:
#         return [-10, -10, -10]







# def vector_field_func(x, y, z):
#     if 0 <= x < 20:
#         # Source: vectors pointing outward from the center (10, 0, 10)
#         # Eigenvalues: (5, 5, 5)
#         return [5 * (x - 10), 5 * y, 5 * (z - 10)]
#
#     elif 20 <= x < 40:
#         # Sink: vectors pointing inward towards the center (30, 0, 10)
#         # Eigenvalues: (-5, -5, -5)
#         return [-5 * (x - 30), -5 * y, -5 * (z - 10)]
#
#     elif 40 <= x < 60:
#         # Saddle: inward in x, outward in z (50, 0, 10)
#         # Eigenvalues: (-10, -10, 10)
#         return [-10 * (x - 50), 0, 10 * (z - 10)]
#
#     elif 60 <= x < 80:
#         # Spiral source: outward spiral in x and y, outward in z (70, 0, 10)
#         # Eigenvalues: (5 + i, 5 - i, 5)
#         dx = 5 * (x - 70) - y  # Rotation in xy-plane
#         dy = 5 * y + (x - 70)
#         dz = 5 * (z - 10)
#         return [dx, dy, dz]
#
#     elif 80 <= x < 100:
#         # Spiral sink: inward spiral in x and y, inward in z (90, 0, 10)
#         # Eigenvalues: (-5 + i, -5 - i, -5)
#         dx = -5 * (x - 90) - y
#         dy = -5 * y + (x - 90)
#         dz = -5 * (z - 10)
#         return [dx, dy, dz]
#
#     elif 100 <= x < 120:
#         # Spiral saddle: inward spiral in x and y, outward in z (110, 0, 10)
#         # Eigenvalues: (-10 + i, -10 - i, 10)
#         dx = -10 * (x - 110) - y
#         dy = -10 * y + (x - 110)
#         dz = 10 * (z - 10)
#         return [dx, dy, dz]
#
#     elif 120 <= x < 140:
#         # Saddle: inward in x, outward in y and z (130, 0, 10)
#         # Eigenvalues: (-5, 5, 5)
#         return [-5 * (x - 130), 5 * y, 5 * (z - 10)]
#
#     else:
#         # Default vector field for undefined regions
#         return [-10, -10, -10]



# def vector_field_func(x, y, z):
#     if 0 <= x < 10 and 0 <= y < 10:
#         # Source: vectors pointing outward from the center (5, 5, 10)
#         # Eigenvalues: (5, 5, 5)
#         return [5 * (x - 5), 5 * (y - 5), 5 * (z - 10)]
#
#     elif 10 <= x < 20 and 0 <= y < 10:
#         # Source: vectors pointing inward towards the center (15, 5, 10)
#         # Eigenvalues: (10, 10, 10)
#         return [-10 * (x - 15), -10 * (y - 5), -10 * (z - 10)]
#
#     elif 20 <= x < 30 and 0 <= y < 10:
#         # Source: (25, 5, 10)
#         # Eigenvalues: (10, 10, 10)
#         return [10 * (x - 25), 10 * (y - 5), 10 * (z - 10)]
#
#     elif 30 <= x < 40 and 0 <= y < 10:
#         # Source: (35, 5, 10)
#         # Eigenvalues: (1, 5, 10)
#         return [1 * (x - 35), 5 * (y - 5), 10 * (z - 10)]
#
#     elif 0 <= x < 10 and 10 <= y < 20:
#         # Sink: (5, 15, 10)
#         # Eigenvalues: (-1, -1, -1)
#         return [-1 * (x - 5), -1 * (y - 15), -1 * (z - 10)]
#
#     elif 10 <= x < 20 and 10 <= y < 20:
#         # Sink: (15, 15, 10)
#         # Eigenvalues: (-5, -5, -5)
#         return [-5 * (x - 15), -5 * (y - 15), -5 * (z - 10)]
#
#     elif 20 <= x < 30 and 10 <= y < 20:
#         # Sink: (25, 15, 10)
#         # Eigenvalues: (-10, -10, -10)
#         return [-10 * (x - 25), -10 * (y - 15), -10 * (z - 10)]
#
#     elif 30 <= x < 40 and 10 <= y < 20:
#         # Sink: (35, 15, 10)
#         # Eigenvalues: (-1, -5, -10)
#         return [-1 * (x - 35), -5 * (y - 15), -10 * (z - 10)]
#
#     elif 0 <= x < 10 and 20 <= y < 30:
#         # Saddle: (5, 25, 10)
#         # Eigenvalues: (5, 5, -5)
#         return [5 * (x - 5), 5 * (y - 25), -5 * (z - 10)]
#
#     elif 10 <= x < 20 and 20 <= y < 30:
#         # Saddle: (15, 25, 10)
#         # Eigenvalues: (5, -5, -5)
#         return [5 * (x - 15), -5 * (y - 25), -5 * (z - 10)]
#
#     elif 20 <= x < 30 and 20 <= y < 30:
#         # Saddle: (25, 25, 10)
#         # Eigenvalues: (-5, 5, 10)
#         return [-5 * (x - 25), 5 * (y - 25), 10 * (z - 10)]
#
#     elif 30 <= x < 40 and 20 <= y < 30:
#         # Saddle: (35, 25, 10)
#         # Eigenvalues: (-10, 10, 10)
#         return [-10 * (x - 35), 10 * (y - 25), 10 * (z - 10)]
#
#     elif 0 <= x < 10 and 30 <= y < 40:
#         # Saddle: (5, 35, 10)
#         # Eigenvalues: (-10, -10, 10)
#         return [-10 * (x - 5), -10 * (y - 35), 10 * (z - 10)]
#
#     elif 10 <= x < 20 and 30 <= y < 40:
#         # Saddle: (15, 35, 10)
#         # Eigenvalues: (-5, 5, -10)
#         return [-5 * (x - 15), 5 * (y - 35), -10 * (z - 10)]
#
#     else:
#         return [-10, -10, -10]
#




# output eigenvalues
# Define grid dimensions and properties
# dims = (80, 40, 20)
dims = (60, 60, 60)
# dims = (20, 20, 20)
spacing = (1.0, 1.0, 1.0)
origin = (0, 0, 0)

# Create the VTK image data with the specified vector field
image_data = create_vtk_image_data(dims, origin, spacing, vector_field_func)

# Write to a .vti file for visualization
writer = vtk.vtkXMLImageDataWriter()
filename = "vector_field_complex_single.vti"
# filename = "vector field data/-5_-5_-5.vti"
# filename = "vector_field_complex.vti"
# filename = "vector_field_complex_vortex_corelines.vti"
# filename = "vector_field_complex_vortex_corelines2.vti"
# filename = "vector_field_complex_critical_points.vti"


writer.SetFileName(filename)
writer.SetInputData(image_data)
writer.Write()

print(f"Vector field saved to {filename}")










