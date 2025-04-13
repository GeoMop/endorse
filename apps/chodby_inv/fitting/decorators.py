from functools import wraps
from time import time

import pandas as pd

from apps.chodby_inv.fitting.measurement_processing import work_dir


def save_to_excel_decorator(func):
    counter = 0  # counter to keep track of how many files were written

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal counter
        # Call the original function and capture its output
        kwargs_ = {k: v for k, v in kwargs.items() if k != "save_to_excel"}
        excel_data = func(*args, **kwargs_)

        # Check if the keyword argument 'save_to_excel' is True
        if kwargs.get("save_to_excel"):
            counter += 1
            # Create the output file name using a two-digit counter and the function name
            output_file = work_dir/f"{counter:02d}_{func.__name__}.xlsx"

            # Write each dataframe to a separate sheet
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                for sheet_name, df in excel_data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

            print(f'Výstup uložen do "{output_file}".')

        return excel_data

    return wrapper


def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        duration = time() - start
        print(f"⏱️ Funkce '{func.__name__}' trvala {duration:.2f} sekund.")
        return result
    return wrapper


def log_function_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"▶️ Volání: {func.__name__}()")
        result = func(*args, **kwargs)
        print(f"✅ Hotovo: {func.__name__}()")
        return result
    return wrapper


def compose_decorators(*decorators):
    """
    Compose multiple decorators into a single decorator.
    Decorators are applied in the order they are passed.
    """
    def composed(func):
        for decorator in reversed(decorators):
            func = decorator(func)
        return func
    return composed


common_report = compose_decorators(log_function_call, measure_time, save_to_excel_decorator)
