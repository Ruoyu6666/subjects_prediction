import os
from dotenv import load_dotenv
from utils import *
from gnn import *


load_dotenv()


CONTENT_PATH = os.getenv('CONTENT_PATH')
CITES_PATH = os.getenv('CITES_PATH')
PREDICTIONS_PATH = os.getenv('PREDICTIONS_PATH')
