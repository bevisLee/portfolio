import numpy as np

L = list(range(10))
L

L2 = [str(c) for c in L]
L2

type(L2[0])

L3 = [True, "2", 3.0, 4]
L3
[type(item) for item in L3]

import array
L = list(range(10))
A = array.array('i', L)
A

# integer array:
np.array([1, 4, 2, 5, 3])

np.array([3.14, 4, 2, 3])

np.array([1, 2, 3, 4], dtype='float32')

# nested lists result in multi-dimensional arrays
np.array([range(i, i + 3) for i in [2, 4, 6]])

# Create a length-10 integer array filled with zeros
np.zeros(10, dtype=int)

# Create a 3x5 floating-point array filled with ones
np.ones((3, 5), dtype=float)

# Create a 3x5 array filled with 3.14
np.full((3, 5), 3.14)

# Create an array filled with a linear sequence
# Starting at 0, ending at 20, stepping by 2
# (this is similar to the built-in range() function)
np.arange(0, 20, 2)

# Create an array of five values evenly spaced between 0 and 1
np.linspace(0, 1, 5)

# Create a 3x3 array of uniformly distributed
# random values between 0 and 1
np.random.random((3, 3))

# Create a 3x3 array of normally distributed random values
# with mean 0 and standard deviation 1
np.random.normal(0, 1, (3, 3))

# Create a 3x3 array of random integers in the interval [0, 10)
np.random.randint(0, 10, (3, 3))

# Create a 3x3 identity matrix
np.eye(3)

# Create an uninitialized array of three integers
# The values will be whatever happens to already exist at that memory location
np.empty(3)

import numpy as np
np.random.seed(0)  # seed for reproducibility / 동일한 값이 출력되도록 고정

x1 = np.random.randint(10, size=6)  # One-dimensional array
x2 = np.random.randint(10, size=(3, 4))  # Two-dimensional array
x3 = np.random.randint(10, size=(3, 4, 5))  # Three-dimensional array

print("x3 ndim: ", x3.ndim)
print("x3 shape:", x3.shape)
print("x3 size: ", x3.size)

print("dtype:", x3.dtype)

print("itemsize:", x3.itemsize, "bytes")
print("nbytes:", x3.nbytes, "bytes")

x1

x1[0]

x1[4]

x1[-1]

x1[-2]

x2

x2[0, 0]

x2[2, 0]

x2[2, -1]

x2[0, 0] = 12
x2

x1[0] = 3.14159  # this will be truncated!
x1

x = np.arange(10)
x

x[:5]  # first five elements

x[5:]  # elements after index 5

x[4:7]  # middle sub-array

x[::2]  # every other element

x[1::2]  # every other element, starting at index 1

x[::-1]  # all elements, reversed

x[5::-2]  # reversed every other from index 5

x2




