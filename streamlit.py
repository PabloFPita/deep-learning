# Create a streamlit app to display the results of the model 

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('data.csv')
