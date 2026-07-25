"""
Phase 1b — Data Contract Enforcement with Great Expectations.
Run: python3 src/validate_data.py data/spark_cleaned_churn.csv
"""

import sys
import pandas as pd
import great_expectations as gx


def build_expectation_suite() -> gx.ExpectationSuite:
    suite = gx.ExpectationSuite(name="churn_data_contract")

    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="customer_id"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeUnique(column="customer_id"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="customer_id"))

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(column="tenure", min_value=0, max_value=100)
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(column="monthly_charges", min_value=0, max_value=500)
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(column="total_charges", min_value=0, max_value=15000)
    )
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="total_charges"))

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="contract_type", value_set=["Month-to-month", "One year", "Two year"]
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="internet_service", value_set=["DSL", "Fiber optic", "No"]
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="payment_method",
            value_set=[
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)",
            ],
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(column="senior_citizen", value_set=[0, 1])
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(column="churn", value_set=[0, 1])
    )
    return suite


def validate(csv_path: str) -> bool:
    df = pd.read_csv(csv_path)
    context = gx.get_context(mode="ephemeral")

    data_source = context.data_sources.add_pandas("churn_pandas_source")
    data_asset = data_source.add_dataframe_asset(name="churn_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("full_batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = build_expectation_suite()
    result = batch.validate(suite)

    print(f"\n{'='*60}\nVALIDATION {'PASSED' if result.success else 'FAILED'}\n{'='*60}")
    for r in result.results:
        expectation_type = r["expectation_config"]["type"]
        column = r["expectation_config"]["kwargs"].get("column", "N/A")
        status = "PASS" if r["success"] else "FAIL"
        print(f"  [{status}] {expectation_type} (column: {column})")
        if not r["success"]:
            print(f"         -> {r['result'].get('unexpected_count', '?')} unexpected values found")

    return result.success


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw_telco_churn.csv"
    success = validate(path)
    if not success:
        print("\nData contract violated. Halting pipeline before training.")
        sys.exit(1)
    print("\nData contract satisfied. Safe to proceed to training.")