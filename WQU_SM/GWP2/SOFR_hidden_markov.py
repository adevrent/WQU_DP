import pandas as pd
import numpy as np
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the CSV file
csv_file_path = os.path.join(script_dir, 'SOFR.csv')

# Read the CSV file
df = pd.read_csv(csv_file_path)
print(df.head())