import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,current_dir)
# print(sys.path)
from docloader import *
from imgloader import *
from pdfloader import *
from pptloader import *