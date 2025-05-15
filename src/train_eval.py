import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import tensorflow as tf
import networkx as nx
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, Model
from utils import *
from gnn import *


load_dotenv()


CONTENT_PATH = os.getenv('CONTENT_PATH')
CITES_PATH = os.getenv('CITES_PATH')
PREDICTIONS_PATH = os.getenv('PREDICTIONS_PATH')
