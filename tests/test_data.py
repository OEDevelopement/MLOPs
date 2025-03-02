# Datenqualitätstests
# BEISPIEL INPUT

import great_expectations as ge
import pandas as pd

def test_data_quality():
    df = pd.read_csv("data/train.csv")
    df_ge = ge.from_pandas(df)

    expectation = df_ge.expect_column_values_to_not_be_null("feature1")
    assert expectation["success"] == True