import great_expectations as gx
import pandas as pd

df = pd.read_csv('mlflow/data/processed/processed_data.csv')

context = gx.get_context()
data_source = context.data_sources.add_pandas(name = "my_pandas_datasource")
data_asset = data_source.add_dataframe_asset(name = "my_dataframe_asset")
batch_definition = data_asset.add_batch_definition_whole_dataframe(name = "my_batch_definition")
batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

age_expect = gx.expectations.ExpectColumnValuesToBeOfType(
    column='age',
    type_='int'
)

cap_expect = gx.expectations.ExpectColumnValuesToBeInSet(
    column='gained-capital',
    value_set=[0, 1]
)

workclass_expect = gx.expectations.ExpectColumnValuesToBeInSet(
    column='workclass',
    value_set=['Government', 'Self Employed', 'Unemployed', 'Private']
)

education_expect = gx.expectations.ExpectColumnValuesToBeOfType(
    column='educational-num',
    type_='int'
)

marital_status_expect = gx.expectations.ExpectColumnValuesToBeInSet(
    column='marital-status',
    value_set=['Widowed/Separated', 'Married', 'Never-married']
)

occupation_expect = gx.expectations.ExpectColumnValuesToBeInSet(
    column='occupation',
    value_set=['Simple Services', 'Public Safety', 'Specialized Services', 'Professional', 'Management', 'Administrative', 'Sales']
)

relationship_expect = gx.expectations.ExpectColumnValuesToBeInSet(
    column='relationship',
    value_set=['Shared Housing', 'Child', 'Husband', 'Wife', 'Single']
)

race_expect = gx.expectations.ExpectColumnValuesToBeInSet(
    column='is_White',
    value_set=[0, 1]
)

gender_expect = gx.expectations.ExpectColumnValuesToBeInSet(
    column='is_Male',
    value_set=[0, 1]
)

native_region_expect = gx.expectations.ExpectColumnValuesToBeInSet(
    column='from_USA',
    value_set=[0, 1]
)

income_expect = gx.expectations.ExpectColumnValuesToBeInSet(
    column='income >50K',
    value_set=[0, 1]
)

suite = gx.ExpectationSuite(name = "my_suite")
suite.expectations.extend([age_expect, cap_expect, workclass_expect,
                           education_expect, marital_status_expect, occupation_expect,
                           relationship_expect, race_expect, gender_expect,
                           native_region_expect, income_expect])

validation_results = batch.validate(expect=suite)

print(validation_results.success)