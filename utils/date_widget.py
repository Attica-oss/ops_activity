import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import anywidget
    import traitlets
    import marimo as mo
    from datetime import datetime,timedelta
    from typing import List, Union


@app.class_definition
class DateDropdownWidget(anywidget.AnyWidget):
    # Widget front-end JavaScript code
    _esm = """
    function render({ model, el }) {
        // Create container div
        let container = document.createElement("div");
        container.className = "date-dropdown-container";

        // Create label
        let label = document.createElement("label");
        label.innerHTML = "Select Date: ";
        label.className = "date-label";

        // Create dropdown select
        let select = document.createElement("select");
        select.className = "date-select";

        // Function to populate dropdown options
        function populateOptions() {
            // Clear existing options
            select.innerHTML = "";

            // Add default option
            let defaultOption = document.createElement("option");
            defaultOption.value = "";
            defaultOption.innerHTML = "-- Select a date --";
            defaultOption.disabled = true;
            defaultOption.selected = model.get("selected_date") === "";
            select.appendChild(defaultOption);

            // Get available dates from model
            let availableDates = model.get("available_dates");

            // Add date options
            availableDates.forEach(date => {
                let option = document.createElement("option");
                option.value = date;
                option.innerHTML = date;
                option.selected = date === model.get("selected_date");
                select.appendChild(option);
            });
        }

        // Initial population
        populateOptions();

        // Handle selection change
        select.addEventListener("change", (event) => {
            model.set("selected_date", event.target.value);
            model.save_changes();
        });

        // Listen for changes to available dates
        model.on("change:available_dates", () => {
            populateOptions();
        });

        // Listen for changes to selected date (from Python side)
        model.on("change:selected_date", () => {
            // Update selected option
            Array.from(select.options).forEach(option => {
                option.selected = option.value === model.get("selected_date");
            });
        });

        // Append elements to container
        container.appendChild(label);
        container.appendChild(select);

        // Append container to widget element
        el.appendChild(container);
    }
    export default { render };
    """

    _css = """
    .date-dropdown-container {
        display: flex;
        flex-direction: column;
        gap: 8px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        max-width: 300px;
    }

    .date-label {
        font-weight: 500;
        font-size: 14px;
        color: #333;
        margin: 0;
    }

    .date-select {
        padding: 8px 12px !important;
        border: 2px solid #e1e5e9 !important;
        border-radius: 6px !important;
        background-color: white !important;
        font-size: 14px !important;
        color: #333 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        outline: none !important;
    }

    .date-select:hover {
        border-color: #c1c7cd !important;
        background-color: #f8f9fa !important;
    }

    .date-select:focus {
        border-color: #0066cc !important;
        box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.1) !important;
        background-color: white !important;
    }

    .date-select:disabled {
        background-color: #f5f5f5 !important;
        color: #999 !important;
        cursor: not-allowed !important;
        border-color: #e1e5e9 !important;
    }

    .date-select option {
        padding: 4px !important;
        font-size: 14px !important;
    }

    .date-select option:disabled {
        color: #999 !important;
        font-style: italic !important;
    }
    """

    # Widget properties
    available_dates = traitlets.List(trait=traitlets.Unicode()).tag(sync=True)
    selected_date = traitlets.Unicode("").tag(sync=True)

    def __init__(self, dates: List[Union[str, datetime]] = None, **kwargs):
        """
        Initialize the DateDropdownWidget.

        Parameters:
        -----------
        dates : List[Union[str, datetime]], optional
            List of available dates. Can be datetime objects or string dates.
            If datetime objects, they will be formatted as YYYY-MM-DD.
        """
        super().__init__(**kwargs)

        if dates:
            self.set_available_dates(dates)

    def set_available_dates(self, dates: List[Union[str, datetime]]):
        """
        Set the available dates for the dropdown.

        Parameters:
        -----------
        dates : List[Union[str, datetime]]
            List of available dates. Can be datetime objects or string dates.
        """
        formatted_dates = []

        for date in dates:
            if isinstance(date, datetime):
                formatted_dates.append(date.strftime("%Y-%m-%d"))
            elif isinstance(date, str):
                formatted_dates.append(date)
            else:
                # Try to convert to string
                formatted_dates.append(str(date))

        # Sort dates
        try:
            formatted_dates.sort(key=lambda x: datetime.strptime(x, "%Y-%m-%d"))
        except ValueError:
            # If dates aren't in YYYY-MM-DD format, sort as strings
            formatted_dates.sort()

        self.available_dates = formatted_dates

    def get_selected_date(self) -> str:
        """Get the currently selected date."""
        return self.selected_date

    def set_selected_date(self, date: Union[str, datetime]):
        """
        Set the selected date programmatically.

        Parameters:
        -----------
        date : Union[str, datetime]
            The date to select.
        """
        if isinstance(date, datetime):
            date_str = date.strftime("%Y-%m-%d")
        else:
            date_str = str(date)

        if date_str in self.available_dates:
            self.selected_date = date_str
        else:
            raise ValueError(f"Date '{date_str}' is not in the list of available dates")


@app.cell
def _():
    base_date = datetime(2024, 1, 1)

    sample_dates = [base_date + timedelta(days=i*7) for i in range(10)]  # Every week for 10 weeks

        # Create the widget
    date_widget = mo.ui.anywidget(DateDropdownWidget(dates=sample_dates))
    return (date_widget,)


@app.cell
def _(date_widget):
    date_widget
    return


@app.cell
def _(date_widget):
    date_widget.selected_date
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
