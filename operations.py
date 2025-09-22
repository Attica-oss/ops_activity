import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    import polars as pl
    from utils import google_sheet as gs
    from utils.date_widget import DateDropdownWidget
    from datetime import date,datetime
    import logging

    from typing import Optional


@app.cell
def logger():
    # Configure logging for better error tracking
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return (logger,)


@app.cell
def loadgooglesheetdata(logger):
    # Initialize the Google Sheets loader with error handling
    try:
        gsheet = gs.GoogleSheetsLoader()
        operations_raw_data = gsheet.load_sheet(config_name="Operations_Activity").data
        customers = gsheet.load_sheet(config_name="customer").data
        transfer = gsheet.load_sheet(config_name="Transfer").data
        logger.info("Successfully loaded data from Google Sheets")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        mo.stop("❌ Failed to load data from Google Sheets. Please check your connection.")
    return customers, operations_raw_data, transfer


@app.cell
def _():
    # Unique containers


    def transfer_clean(df: pl.LazyFrame) -> pl.LazyFrame:
        """Transfer dataset"""
        return df.select(pl.col("container_number"), pl.col("line")).unique()


    def iot_soc(df: pl.LazyFrame) -> pl.LazyFrame:
        """IOT SOC containers"""
        return df.filter(pl.col("line").eq("IOT")).drop("line")


    def container_stuffing(df: pl.LazyFrame) -> pl.LazyFrame:
        """Container numbers"""
        return df.filter(pl.col("line").ne("IOT")).drop("line")
    return container_stuffing, iot_soc, transfer_clean


@app.cell
def _(container_stuffing, iot_soc, transfer, transfer_clean):
    iot_soc_unique = iot_soc(transfer_clean(df=transfer)).collect().to_series().to_list()
    container_unique = container_stuffing(transfer_clean(df=transfer)).collect().to_series().to_list()
    return container_unique, iot_soc_unique


@app.cell
def _(logger):
    def clean_ops_data(df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Clean and transform operations data with enhanced error handling

        Args:
            df: Raw operations data from Google Sheets

        Returns:
            Cleaned LazyFrame with standardized columns and data types
        """
        try:
            return df.select(
                pl.col("Day"),
                pl.col("Date").alias("date").cast(pl.Date, strict=False),
                pl.col("Time").alias("time").cast(pl.Time, strict=False),
                pl.col("Vessel").str.to_uppercase().alias("vessel"),
                pl.col("Species").str.extract(r"^(.*?)(\s-\s|$)").alias("species"),
                pl.col("Details").str.to_uppercase().alias("details"),
                # Enhanced tonnage calculation with better error handling
                (pl.col("Scale Reading(-Fish Net) (Cal)")
                 .str.replace_all(",", "")
                 .str.replace_all(r"[^\d.]", "")  # Remove non-numeric characters
                 .cast(pl.Float64, strict=False)
                 .fill_null(0)
                 .alias("tonnage_kg")
                 * 0.001).round(3).alias("tonnage"),
                pl.col("Storage").cast(dtype=pl.Enum(["Brine", "Dry"]), strict=False),
                pl.col("Container (Destination)").str.to_uppercase().alias("destination"),
                pl.col("overtime"),
                pl.col("Side Working").alias("side_working")
            ).filter(
                # Filter out invalid records
                pl.col("date").is_not_null() &
                pl.col("vessel").is_not_null()
                # pl.col("tonnage").ge(0)
            )
        except Exception as e:
            logger.error("Error cleaning operations data: %s",e)
            raise  pl.col("Side Working").alias("side_working")
    return (clean_ops_data,)


@app.cell
def _(clean_ops_data, operations_raw_data):
    operation_dataset = clean_ops_data(operations_raw_data).collect()
    return (operation_dataset,)


@app.function
def clean_customers(df: pl.LazyFrame, vessel_kind: list[str]) -> list[str]:
    """Clean the customer dataset"""
    return (
        df.filter(pl.col("Type").is_in(vessel_kind))
        .select(pl.col("Vessel/Client"))
        .collect()
        .to_series()
        .to_list()
    )


@app.cell
def _(logger):

    def date_string_to_date(date_string: str) -> Optional[date]:
        """
        Convert date string to date object with multiple format support

        Args:
            date_string: Date string in various formats

        Returns:
            date object or None if conversion fails
        """
        if not date_string:
            return None

        formats = ["%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"]

        for fmt in formats:
            try:
                return datetime.strptime(date_string, fmt).date()
            except ValueError:
                continue

        logger.warning(f"Could not parse date string: {date_string}")
        return None
    return (date_string_to_date,)


@app.cell
def _(customers):
    # Create enhanced vessel dropdown
    vessel_options = clean_customers(customers, ["THONIER", "LONGLINER", "CARGO"])

    if not vessel_options:
        mo.stop("❌ No vessels found in the customer data")

    vessel_dropdown = mo.ui.dropdown(
        options=vessel_options,
        searchable=True,
        label="🚢 Select Vessel"
    )

    # Enhanced UI layout
    title = mo.md("# 📊 Hourly Operations Analysis")
    sub_title = mo.md("Select a vessel to view hourly tonnage performance")

    mo.vstack([title,sub_title,vessel_dropdown])
    return (vessel_dropdown,)


@app.cell
def _(vessel_dropdown):




    # Stop execution if no vessel is selected
    if vessel_dropdown.value is None:
        mo.stop("👆 Please select a vessel to continue")

    selected_vessel = vessel_dropdown.value
    return (selected_vessel,)


@app.cell
def _(operation_dataset, selected_vessel):
    # Create hourly summary
    try:
        hourly = create_operation_summary(operation_dataset)
        filtered_data = hourly.filter(pl.col("vessel") == selected_vessel)

        if len(filtered_data) == 0:
            mo.stop(f"❌ No data found for vessel: {selected_vessel}")

        # Get available dates with proper formatting
        available_dates = (filtered_data
            .select(pl.col("date").dt.to_string("%d/%m/%Y").unique())
            .sort(pl.col("date"))
            .to_series()
            .to_list()
        )

    except Exception as e:
        mo.stop(f"❌ Error processing data for {selected_vessel}: {e}")

    # Enhanced date dropdown
    date_dropdown = mo.ui.dropdown(
        label=f"📅 Available dates for {selected_vessel} ({len(available_dates)} days)",
        options=available_dates
    )

    date_dropdown
    return (date_dropdown,)


@app.function
def create_operation_summary(operation_dataset):
    # First, get the pivoted data
    base_query = (operation_dataset
        .with_columns(pl.col("time").dt.hour().add(1).alias("hour"))
        .group_by(pl.col("date"),pl.col("vessel"),pl.col("hour"), pl.col("side_working"), maintain_order=True)
        .agg(pl.col("tonnage").sum().round(3))
        .pivot("side_working", index=["date","vessel","hour"], values="tonnage", aggregate_function="sum")
        .sort(by=["date","hour"])
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
def _(date_dropdown):
    # Stop if no date selected
    if date_dropdown.value is None:
        mo.stop("👆 Please select a date to view hourly data")

    # Enhanced target slider with better UI
    target_per_hour = mo.ui.slider(
        label="🎯 Target Tonnage Per Hour",
        start=15,
        stop=40,
        step=1,
        value=25,
        show_value=True
    )

    mo.vstack([mo.md("### Performance Settings"),
    target_per_hour])
    return (target_per_hour,)


@app.cell
def _(operation_dataset, target_per_hour):
    hourly_ = create_operation_summary(operation_dataset=operation_dataset).with_columns((pl.col("tonnage_per_hour")-pl.lit(target_per_hour.value)).round(1).alias("🎯"))
    return (hourly_,)


@app.cell
def _(date_dropdown, date_string_to_date, hourly_, vessel_dropdown):
    hourly_filtered = hourly_.filter(
        pl.col("date").eq(date_string_to_date(date_dropdown.value)),
        pl.col("vessel").eq(vessel_dropdown.value),
    ).drop(["date", "vessel"]).with_columns(total_tonnage=pl.col("tonnage_per_hour").cum_sum().round(3))

    mo.ui.table(hourly_filtered)
    return


@app.cell
def _(date_dropdown, date_string_to_date, operation_dataset, vessel_dropdown):
    import polars.selectors as cs

    daily_ = operation_dataset.filter(
        pl.col("date").eq(date_string_to_date(date_dropdown.value)),
        pl.col("vessel").eq(vessel_dropdown.value),
    ).group_by(
        pl.col("destination"),
        pl.col("species"),
        pl.col("Storage"),
        maintain_order=True,
    ).agg(pl.col("tonnage").sum().round(3)).with_columns(
        species=pl.col("species") + "\n" + pl.col("Storage").cast(pl.Utf8)
    ).drop("Storage").pivot(
        "species", index="destination", values="tonnage", aggregate_function="sum"
    ).with_columns(total=pl.sum_horizontal(cs.numeric()).round(3))
    return (daily_,)


@app.cell
def _(daily_, date_dropdown, vessel_dropdown):
    date_daily = date_dropdown.value
    vessel_daily = vessel_dropdown.value


    mo.vstack(
        [
            mo.md("# Daily Unloading Report"),
            f"Vessel: {vessel_daily}",
            f"Date: {date_daily}",
            mo.ui.table(daily_),
            mo.md(f"## Total: {daily_.select(pl.col("total").sum().round(3)).to_series().to_list()[0]}")
        ]
    )
    return


@app.cell
def _(customers):
    cargo_list = clean_customers(customers, ["CARGO"])
    return (cargo_list,)


@app.cell
def _(cargo_list, container_unique, iot_soc_unique, operation_dataset):
    def operation_with_kind() -> pl.DataFrame:
        return operation_dataset.with_columns(
            operation_kind=pl.when(pl.col("destination").is_in(cargo_list))
            .then(pl.lit("Transhipment"))
            .when(
                (pl.col("destination").is_in(iot_soc_unique)).or_(
                    pl.col("destination")
                    .eq(pl.lit("UNLOAD TO QUAY"))
                    .or_(
                        pl.col("destination").is_in(
                            ["CCCS (IOT)", "CCCS (DARDANEL)"]
                        )
                    )
                )
            )
            .then(pl.lit("Simple Unloading"))
            .when(pl.col("destination").str.contains(pl.lit("CCCS")))
            .then(pl.lit("Unload to CCCS"))
            .when(pl.col("destination").is_in(container_unique))
            .then(pl.lit("Container Stuffing"))
            .otherwise(pl.col("destination"))
        )
    return (operation_with_kind,)


@app.cell
def _(operation_with_kind):
    summarised_ops = operation_with_kind().with_columns(operation_kind=pl.col("operation_kind")+"\n"+pl.col("Storage").cast(pl.Utf8)).group_by(
        pl.col("operation_kind"),
        # pl.col("Storage"),
        pl.col("date"),
        pl.col("vessel"),
        maintain_order=True,
    ).agg(pl.col("tonnage").sum().round(3)).pivot("operation_kind",index=["date","vessel"],values="tonnage",aggregate_function="sum")
    return (summarised_ops,)


@app.cell
def _(summarised_ops):
    mo.vstack([mo.md("# Operations Activity"),mo.ui.table(summarised_ops)])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
