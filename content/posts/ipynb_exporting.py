import torch
import numpy as np

one = torch.ones(2,3)
x = torch.empty(4,3)
x * 23

torch.manual_seed(42)
many = torch.rand(3,2,1)
many

n1 = np.array([[1,3,4,5],[7,5,[1,2,3]]])

np.array(['vivec',[1, 'two', 3]])

np.array([1, 'two', 3])
torch.rand(1, 'two', 3)

torch.zeros(5,4,3,3)

torch.ones(3,3,3,3)

torch.ones(2,2,2,3,5).shape

import nbformat
from pathlib import Path
Path.home().exists()
Path('./').absolute()
.exists()

import json
nb_location = Path('./content/posts/tensor-refresher/fi.ipynb').absolute()
nb_location.exists()
nb_location

with open(nb_location, 'r') as read_file:
    data = json.load(read_file, )

nb = nbformat.reads(json.dumps(data), as_version=4)
nb.cells[0].keys()

from traitlets.config import Config
from nbconvert import MarkdownExporter, HTMLExporter
import nbconvert

HTMLExporter(template_name = 'Classic')
e = MarkdownExporter()
out = e.from_notebook_node(nb=nb)
Path()
o = Path('./').absolute().joinpath('out.md')
with open(str(o), 'w') as write_file:
    write_file.write(str(out))