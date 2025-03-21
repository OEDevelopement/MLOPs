import great_expectations as gx
import pandas as pd

df = pd.read_csv('./mlflow/data/raw/adult.csv')

context = gx.get_context()
data_source = context.data_sources.add_pandas(name = "my_pandas_datasource")
data_asset = data_source.add_dataframe_asset(name = "my_dataframe_asset")
batch_definition = data_asset.add_batch_definition_whole_dataframe(name = "my_batch_definition")
batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

colnames = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status', 'occupation', 
            'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 
            'income']

int_cols = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']

binary_cols = ['native-country', 'income', 'race', 'gender']

workclass_vals = ['Private', 'Self-emp-not-inc', 'Local-gov', '?', 'State-gov', 'Self-emp-inc',
                  'Federal-gov', 'Without-pay', 'Never-worked']

marital_status_vals = ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 
                       'Married-spouse-absent', 'Married-AF-spouse']

occupation_vals = ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 
                   'Other-service', 'Machine-op-inspct', '?', 'Transport-moving', 'Handlers-cleaners', 
                   'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces']

relationship_vals = ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative']

# Expectation Suite erstellen
suite = gx.ExpectationSuite(name = "my_suite")

# 1. Für alle Spalten überprüfen, ob sie existieren
for col in colnames:
    suite.expectations.extend(
        [gx.expectations.ExpectColumnToExist(column=col)]
    )

# 2. Überprüfen, ob die Anzahl der Spalten passt
suite.expectations.extend(
    [gx.expectations.ExpectTableColumnCountToEqual(value=len(colnames))]
)

# 3. Gucken ob alle Spalten keine NA haben
for col in colnames:
    suite.expectations.extend(
        [gx.expectations.ExpectColumnValuesToNotBeNull(column=col)]
    )

# 4. Für alle Integer Spalten gucken ob der Typ Integer ist
for col in int_cols:
    suite.expectations.extend(
        [gx.expectations.ExpectColumnValuesToBeOfType(column=col, type_="int64")]
    )

# 5. Für alle kategorischen Spalten gucken, ob die jeweiligen Ausprägungen einzig vorkommen
# Workclass
suite.expectations.extend(
    [gx.expectations.ExpectColumnDistinctValuesToBeInSet(
        column="workclass", value_set=workclass_vals
    )]
)

# Marital-status
suite.expectations.extend(
    [gx.expectations.ExpectColumnDistinctValuesToBeInSet(
        column="marital-status", value_set=marital_status_vals
    )]
)

# Occupation
suite.expectations.extend(
    [gx.expectations.ExpectColumnDistinctValuesToBeInSet(
        column="occupation", value_set=occupation_vals
    )]
)

# Relationship
suite.expectations.extend(
    [gx.expectations.ExpectColumnDistinctValuesToBeInSet(
        column="relationship", value_set=relationship_vals
    )]
)

# 6. Für die binären Spalten gucken, ob der Wert, der vorkommen muss, darin vorkommt
# Native-country muss USA enthalten
suite.expectations.extend(
    [gx.expectations.ExpectColumnDistinctValuesToContainSet(
        column="native-country", value_set=["United-States"]
    )]
)

# Income muss >50K enthalten
suite.expectations.extend(
    [gx.expectations.ExpectColumnDistinctValuesToContainSet(
        column="income", value_set=[">50K"]
    )]
)

# Race muss White enthalten
suite.expectations.extend(
    [gx.expectations.ExpectColumnDistinctValuesToContainSet(
        column="race", value_set=["White"]
    )]
)

# Gender muss Male enthalten
suite.expectations.extend(
    [gx.expectations.ExpectColumnDistinctValuesToContainSet(
        column="gender", value_set=["Male"]
    )]
)

validation_results = batch.validate(expect=suite)

print(validation_results.success)
