---
title: "Quarto Basics"
date: 2023-04-21
draft: true
format: hugo-md
jupyter: python3
---

``` python
import sys
print(sys.executable)
```

    C:\Users\Welp Windows 10\AppData\Local\Programs\Python\Python311\python.exe

For a demonstration of a line plot on a polar axis, see [Figure 1](#fig-polar).

``` python
import numpy as np
import matplotlib.pyplot as plt

r = np.arange(0, 2, 0.01)
theta = 4 * np.pi * r
fig, ax = plt.subplots(
  subplot_kw = {'projection': 'polar'} 
)
ax.plot(theta, r)
ax.set_rticks([0.5, 1, 1.5, 2])
ax.grid(True)
plt.show()
```

<img src="index_files/figure-markdown_strict/fig-polar-output-1.png" id="fig-polar" width="450" height="439" alt="Figure 1: A line plot on a polar axis" />
