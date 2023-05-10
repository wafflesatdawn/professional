---
title: "Tensor Refresher"
date: 2023-03-01T20:18:09-07:00
format: hugo-md
jupyter: python3
draft: true
---

## Intro

Tensors are the foundational building blocks of machine learning and their nuances are worth spending some time on. This post is meant to be a refresher on pytorch tensors, pulling together information from various sources.

## Creating tensors

``` python
import torch
import math

x = torch.empty(3,4)
print(type(x))
print(x)
```

    <class 'torch.Tensor'>
    tensor([[-1.7006e+03,  8.1556e-43, -1.6574e+03,  8.1556e-43],
            [-2.9176e+03,  8.1556e-43, -1.7958e+03,  8.1556e-43],
            [-2.5965e+33,  0.0000e+00,  1.4013e-45,  0.0000e+00]])

To start off, the [pytorch docs](https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html) include a great primer that begins by initializing a tensor just like the one above:
\> Let's unpack what we just did:
\> - We created a tensor using one of the numerous factory methods attached to the torch module.
\> - The tensor itself is 2-dimensional, having 3 rows and 4 columns.
\> - The type of the object returned is torch.Tensor, which is an alias for torch.FloatTensor; by default, PyTorch tensors are populated with 32-bit floating point numbers. (More on data types below.)
\> - You will probably see some random-looking values when printing your tensor. The torch.empty() call allocates memory for the tensor, but does not initialize it with any values - so what you're seeing is whatever was in memory at the time of allocation.

However, uncontrolled random tensor starting values are often less useful than all zeros or ones. If randomness is desired, it can be fixed to be the same every time using `manual_seed`:

``` python
zeros = torch.zeros(2, 3)
print(zeros)

ones = torch.ones(2, 3)
print(ones)

torch.manual_seed(1729)
random = torch.rand(2, 3)
print(random)
```

    tensor([[0., 0., 0.],
            [0., 0., 0.]])
    tensor([[1., 1., 1.],
            [1., 1., 1.]])
    tensor([[0.3126, 0.3791, 0.3087],
            [0.0736, 0.4216, 0.0691]])

## Comparison with numpy arrays

Tensors are higher dimensional analogues to numpy arrays, sharing many qualities such as their objectives, apis, syntax, and functionality. Notable differences do exist however:
- tensors have specialized gpu processing support to accelerate operations
- except for broadcasting operations, tensors cannot be 'jagged', meaning having dimensions of different length
- all data in a tensor must be the same type
Runtime and syntax errors will appear if these restrictions are ignored

``` python
x + zeros

RuntimeError: The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 1
```

``` python
import numpy as np
np.array([1, 'two', 3])
```

    array(['1', 'two', '3'], dtype='<U11')

``` python
torch.rand([1, 'two', 3])
TypeError: rand(): argument 'size' must be tuple of ints, but found element of type str at pos 2
```

### Going beyond 3d

Higher dimensionality is difficult to visualize and often invites people to start drawing volumes and hypercubes. It's simply a consequence of 'dimension' in common usage as well as a sign of how far the cartesian system has permeated. How do you think beyond 3D? We can go 'back to 2D', in a sense. Forget about trying to imagine the data in shapes, connecting lines between them, and instead consider tensor dimensionaliy simply as arrays of arrays (or lists of lists).

Observe how the output grows when we create tensors of increasing size:

``` python
torch.ones(2,2)
```

    tensor([[1., 1.],
            [1., 1.]])

``` python
torch.ones(2,2,2)
```

    tensor([[[1., 1.],
             [1., 1.]],

            [[1., 1.],
             [1., 1.]]])

``` python
torch.ones(2,2,2,2)
```

    tensor([[[[1., 1.],
              [1., 1.]],

             [[1., 1.],
              [1., 1.]]],


            [[[1., 1.],
              [1., 1.]],

             [[1., 1.],
              [1., 1.]]]])

There are a couple of ways to read the pattern that emerges from playing around with this. These examples kept all the arguments identical but modifying the last one makes it clearer:

``` python
torch.ones(2,3,1,5)
```

    tensor([[[[1., 1., 1., 1., 1.]],

             [[1., 1., 1., 1., 1.]],

             [[1., 1., 1., 1., 1.]]],


            [[[1., 1., 1., 1., 1.]],

             [[1., 1., 1., 1., 1.]],

             [[1., 1., 1., 1., 1.]]]])

The last number determines the size of the innermost elements of the tensor (here, 5) while the first number determines the number of groups at the highest level. So, if set to 1,

Higher dimension tensors have additional
Tensors of rank 2 and 3 are old hat--it's easy to see how the 9 numbers have become 27: . Rank 4 is where it starts to get hairy.

``` python
torch.ones(5)
```

    tensor([1., 1., 1., 1., 1.])

``` python
torch.ones(5,4)
```

    tensor([[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]])

``` python
torch.ones(5,4,3)
```

    tensor([[[1., 1., 1.],
             [1., 1., 1.],
             [1., 1., 1.],
             [1., 1., 1.]],

            [[1., 1., 1.],
             [1., 1., 1.],
             [1., 1., 1.],
             [1., 1., 1.]],

            [[1., 1., 1.],
             [1., 1., 1.],
             [1., 1., 1.],
             [1., 1., 1.]],

            [[1., 1., 1.],
             [1., 1., 1.],
             [1., 1., 1.],
             [1., 1., 1.]],

            [[1., 1., 1.],
             [1., 1., 1.],
             [1., 1., 1.],
             [1., 1., 1.]]])

``` python
torch.ones(5,4,3,2)
```

    tensor([[[[1., 1.],
              [1., 1.],
              [1., 1.]],

             [[1., 1.],
              [1., 1.],
              [1., 1.]],

             [[1., 1.],
              [1., 1.],
              [1., 1.]],

             [[1., 1.],
              [1., 1.],
              [1., 1.]]],


            [[[1., 1.],
              [1., 1.],
              [1., 1.]],

             [[1., 1.],
              [1., 1.],
              [1., 1.]],

             [[1., 1.],
              [1., 1.],
              [1., 1.]],

             [[1., 1.],
              [1., 1.],
              [1., 1.]]],


            [[[1., 1.],
              [1., 1.],
              [1., 1.]],

             [[1., 1.],
              [1., 1.],
              [1., 1.]],

             [[1., 1.],
              [1., 1.],
              [1., 1.]],

             [[1., 1.],
              [1., 1.],
              [1., 1.]]],


            [[[1., 1.],
              [1., 1.],
              [1., 1.]],

             [[1., 1.],
              [1., 1.],
              [1., 1.]],

             [[1., 1.],
              [1., 1.],
              [1., 1.]],

             [[1., 1.],
              [1., 1.],
              [1., 1.]]],


            [[[1., 1.],
              [1., 1.],
              [1., 1.]],

             [[1., 1.],
              [1., 1.],
              [1., 1.]],

             [[1., 1.],
              [1., 1.],
              [1., 1.]],

             [[1., 1.],
              [1., 1.],
              [1., 1.]]]])

There are a couple of patterns we can see at this point, and they come from different directions. First, obviously, the output in the terminal is growing longer. But at the same time, parts of it seem to be shrinking.

## Operations with tensors

As seen above, it's not possible to perform operations on tensors when they have different shapes. "Shape" means having the same number of dimensions and the same number of items in each dimension. For example, these two are not the same shape:

``` python
# tensor(1,2) - tensor(2,1)
```

### Element access
