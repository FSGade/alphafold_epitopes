#!/usr/bin/env python3

#Import section
import proteinsolver
from pathlib import Path
import os
import torch
import torch_geometric
from kmbio import PDB

#second import because of anger
import os
from pathlib import Path

from IPython.display import HTML
from IPython.display import display

import matplotlib.pyplot as plt
import pandas as pd
import proteinsolver
from kmbio import PDB
from kmtools import sci_tools


#Initial test
print("Hello world")

#Initial variables
print(torch.cuda.is_available())
device = torch.device("cpu")
STRUCTURE_FILE = Path("7bwj.pdb")

#Load structure
structure_all = PDB.load(STRUCTURE_FILE)
structure = PDB.Structure(STRUCTURE_FILE.name + "H", structure_all[0].extract('H'))
assert len(list(structure.chains)) == 1

#Load model
os.system("python ./model.py")
batch_size = 1
num_features = 20
adj_input_size = 2
hidden_size = 128
frac_present = 0.5
frac_present_valid = frac_present
info_size= 1024
state_file = Path("e53-s1952148-d93703104.state")
net = Net(
    x_input_size=num_features + 1, adj_input_size=adj_input_size, hidden_size=hidden_size, output_size=num_features
)
net.load_state_dict(torch.load(state_file, map_location=device))
net.eval()
net = net.to(device)
