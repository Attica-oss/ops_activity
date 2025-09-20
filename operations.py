import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    import polars as pl
    from utils import google_sheet as gs


@app.cell
def _():
    # The google sheet loader
    gsheet = gs.GoogleSheetsLoader()
    return (gsheet,)


@app.cell
def _(gsheet):
    operations_raw_data = gsheet.load_sheet(config_name="Operations_Activity").data

    operations_raw_data.collect()
    return (operations_raw_data,)


@app.function
def clean_ops_data(df: pl.LazyFrame) -> pl.LazyFrame:
    """Select columns and cast datatypes"""
    return df.select(
        pl.col("Day"),
        pl.col("Date").alias("date"),
        pl.col("Time").alias("time"),
        pl.col("Vessel").str.to_uppercase().alias("vessel"),
        pl.col("Species").str.extract(r"^(.*?)(\s-\s)"),
        pl.col("Details").str.to_uppercase(),
       (pl.col("Scale Reading(-Fish Net) (Cal)")
        .str.replace(",", "")
        .cast(pl.Int64)
        .alias("tonnage")
        * 0.001).round(3),
        pl.col("Storage").cast(dtype=pl.Enum(["Brine","Dry"])),
        pl.col("Container (Destination)").alias("destination"),
        pl.col("overtime"),
        pl.col("Side Working").alias("side_working")
    )


@app.cell
def _(operations_raw_data):
    operation_dataset = clean_ops_data(operations_raw_data).collect()
    return (operation_dataset,)


@app.cell
def _():
    # Create a form with multiple elements
    form = (
        mo.md('''
        **Hourly**

        {name}

        {date}
    ''')
        .batch(
            name=mo.ui.text(label="vessel"),
            date=mo.ui.date(label="date"),
        )
        .form(show_clear_button=True, bordered=True)
    )
    return (form,)


@app.cell
def _(form):
    form
    return


@app.cell
def _(form):
    vessel = form.value["name"]
    date = form.value["date"]




    return date, vessel


@app.function
def create_operation_summary(operation_dataset, date, vessel):
    # First, get the pivoted data
    base_query = (
        operation_dataset.filter(
            pl.col("date").eq(date).and_(pl.col("vessel").eq(vessel))
        )
        .with_columns(pl.col("time").dt.hour().add(1).alias("hour"))
        .group_by(pl.col("hour"), pl.col("side_working"), maintain_order=True)
        .agg(pl.col("tonnage").sum().round(3))
        .pivot("side_working", index="hour", values="tonnage", aggregate_function="sum")
        .sort(by="hour")
    )
    
    # Get the actual columns after pivot
    actual_columns = base_query.columns
    expected_sides = ["Front", "Rear", "Middle"]
    
    # Create expressions to handle missing columns
    column_expressions = []
    for side in expected_sides:
        if side in actual_columns:
            # Column exists, just fill nulls with 0
            column_expressions.append(pl.col(side).fill_null(0).alias(side))
        else:
            # Column doesn't exist, create it with 0
            column_expressions.append(pl.lit(0).alias(side))
    
    # Apply the column expressions and calculate total
    result = base_query.with_columns(column_expressions).with_columns(
        pl.sum_horizontal(["Front","Middle","Rear"]).round(3).alias("tonnage_per_hour")
    )
    
    return result


@app.cell
def _():
    target_per_hours = 25
    return (target_per_hours,)


@app.cell
def _(date, operation_dataset, target_per_hours, vessel):
    hourly = create_operation_summary(operation_dataset=operation_dataset,date=date,vessel=vessel).with_columns(total_tonnage=pl.col("tonnage_per_hour").cum_sum().round(3)).with_columns(on_target=(pl.col("tonnage_per_hour")-pl.lit(target_per_hours)).round(1))
    return (hourly,)


@app.cell
def _(date, hourly):
    import altair as alt

    faceted_chart = alt.Chart(hourly).mark_line().encode(
                x=alt.X('hour:O', title='Hour'),
                y=alt.Y('on_target:Q', title='Tonnage'),
         
                # column=alt.Column('vessel:N', title='Vessel')
            ).properties(
                width=300,
                height=200,
                title=f"Tonnage Comparison by Vessel - {date}"
            )
    return (faceted_chart,)


@app.cell
def _(faceted_chart):
    faceted_chart
    return


@app.cell
def _(hourly):
    hourly
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
