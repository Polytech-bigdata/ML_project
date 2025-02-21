from utils import *

X_num, X_cat, y, labels = load_heterogeneous_dataset('../data/ref_data.csv', debugging=True)

# print(f"X_num types: {X_num.dtypes}")
# print(f"X_cat types: {X_cat.dtypes}")
# print(f"y type: {y.dtypes}")
#print(f"X_num[6]: {X_num.iloc[6]}")
#print(f"X_cat[6]: {X_cat.iloc[6]}")

print(f"nombre de cellules vides dans X_num: {count_void_data(X_num)}")
X = imputer_variables(X_num, X_cat, debugging=True)
print(f"nombre de cellules vides dans X: {count_void_data(X)}")