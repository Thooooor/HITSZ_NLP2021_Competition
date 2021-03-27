import re
import random
import tarfile
import requests
import numpy as np
import paddle
from paddle.nn import Embedding


def data_preprocess(corpus):
    data_set = []
    