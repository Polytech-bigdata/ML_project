from utils import *

y, X_num, X_cat, labels = load_heterogeneous_dataset('../data/ref_data.csv')

# print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"X_num shape: {X_num.shape}")
print(f"X_cat shape: {X_cat.shape}")
print(f"X_num: {X_num}")
print(f"X_cat: {X_cat}")
print(f"labels: {labels}")