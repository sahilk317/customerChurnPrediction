# transformer_builder.py

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define pipeline
numeric_cols = Pipeline([
    ('scaler', StandardScaler())
])

categorical_cols = Pipeline([
    ('encoder', OneHotEncoder(drop='first', sparse_output=False)),
    ('scaler', StandardScaler())
])

# Define final transformer
transformer = ColumnTransformer([
    ('numeric_cols', numeric_cols, [0, 4, 5, 6, 7, 8, 9]),
    ('categorical_cols', categorical_cols, [1, 2])
], remainder='passthrough')

transformer.set_output(transform='pandas')
