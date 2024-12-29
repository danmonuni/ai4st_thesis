import pickle

import numpy  # noqa: F401
import pandas  # noqa: F401

PATH = "./data/simulation_data.pkl"

with open(PATH, 'rb') as file:
    # Load the dictionary from the pickle file
    retrieved_dict = pickle.load(file)

print(f"The dictionary contains {len(retrieved_dict)} items.")