from src.install_import import install_if_missing

# Check and install required packages
install_if_missing("pandas")
install_if_missing("numpy")
install_if_missing("matplotlib")
install_if_missing("seaborn")
install_if_missing("prophet")
install_if_missing("sklearn.metrics")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.metrics import mean_absolute_error




