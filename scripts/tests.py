from utils import *

X_num, X_cat, y, labels = load_heterogeneous_dataset('../data/ref_data.csv', debugging=True)

X = imputer_variables(X_num, X_cat, debugging=True)


