import pandas as pd
from transformer_builder import transformer
import dill

# Load and prepare training data
df = pd.read_csv("Churn_Modelling.csv")
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

X = df.drop('Exited', axis=1)
transformer.fit(X)

# Save transformer using dill
with open("model.m5", "wb") as f:
    dill.dump(transformer, f)
