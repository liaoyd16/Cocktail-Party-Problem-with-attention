
# import meta
from meta import *

# import utils
# from utils import *

# dataset_report(dataset)

import json

import numpy as np

dic = json.load(open(dataset+"mixed0.json", "r"))

print( np.array(dic['entries'][0][0]).reshape(128, 64) )