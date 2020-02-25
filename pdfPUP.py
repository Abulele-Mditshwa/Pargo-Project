"""
Author: Abulele Mditshwa
Student No: Junior Data Scientist
email: abulele@capeai.com

The objective of this code is to create a pdf document from the SQL queries that is extracted in GetData.py and PlotCollections.py code.
"""

#Import write to PDF modules
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.pagesizes import landscape
from reportlab.platypus import Image
from reportlab.rl_config import defaultPageSize
from reportlab.pdfbase.pdfmetrics import stringWidth
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from io import StringIO
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch, cm
from PIL import Image
from reportlab.lib.utils import ImageReader
from svglib.svglib import svg2rlg


# these are for the fonts
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics import renderPDF, renderPM



from textwrap import wrap
