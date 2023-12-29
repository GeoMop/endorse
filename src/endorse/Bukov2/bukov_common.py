from typing import *
import pickle

from endorse import common
import time
import h5py

import PyPDF2
import img2pdf
import os
from pathlib import Path

def create_combined_pdf(file_list, output_filename):
    """
    Create a combined PDF file from a list of PDF and PNG files.

    Args:
    file_list (list): A list of file paths (PDF and PNG, as Path objects or strings).
    output_filename (str): The filename for the output combined PDF.
    """
    pdf_writer = PyPDF2.PdfWriter()

    for file_path in file_list:
        file_path = Path(file_path)  # Ensure file_path is a Path object

        if file_path.suffix == '.png':
            # Convert PNG to PDF
            pdf_path = file_path.with_suffix('.pdf')
            with open(file_path, "rb") as img_file, open(pdf_path, "wb") as pdf_file:
                pdf_file.write(img2pdf.convert(img_file.read()))

            file_path = pdf_path  # Update file_path to the new PDF file

        # Add the PDF file to the combined PDF
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page_num in range(len(pdf_reader.pages)):
                pdf_writer.add_page(pdf_reader.pages[page_num])

    # Write out the combined PDF
    with open(output_filename, 'wb') as out:
        pdf_writer.write(out)

    # Optionally, clean up temporary PDF files created from PNGs
    for file_path in file_list:
        file_path = Path(file_path)  # Ensure file_path is a Path object
        if file_path.suffix == '.png':
            os.remove(file_path.with_suffix('.pdf'))


def load_cfg(cfg_file):
    workdir = cfg_file.parent
    cfg = common.config.load_config(cfg_file)
    return workdir, cfg


class HDF5Files:
    def __init__(self, file_paths, mode):
        self.file_paths = file_paths
        self.mode = mode
        self.files = []

    def __enter__(self):
        for path in self.file_paths:
            self.files.append(h5py.File(path, self.mode))
        return self.files

    def __exit__(self, exc_type, exc_value, traceback):
        for file in self.files:
            file.close()




def pkl_write(workdir, data, name):
    with open(workdir / name, 'wb') as f:
        pickle.dump(data, f)


def pkl_read(workdir, name):
    try:
        with open(workdir / name, 'rb') as f:
            result = pickle.load(f)
    except Exception:
        result = None
    return result


# def memoize(func):
#     def wrapper(workdir, *args, **kwargs):
#         fname = f"{func.__name__}.pkl"
#         val = pkl_read(workdir, fname)
#         if val is None:
#             val = func(args, kwargs)
#             pkl_write(workdir, val, fname)
#         return val
#     return wrapper
#



def memoize(func):
    """
    Simple memoization function, no dependence on input, store into
    a file derived from the function name.
    :param func:
    :return:
    """
    def wrapper(workdir, *args, **kwargs):
        fname = f"{func.__name__}.pkl"
        val = pkl_read(workdir, fname)
        force = kwargs.pop('force', False)
        if force is True or val is None:
            print(f"Execute {func.__name__}  ...", end='')
            start = time.process_time_ns()
            val = func(workdir, *args, **kwargs)
            sec = (time.process_time_ns() - start) / 1e9
            print(f"[{sec}] s.")

            pkl_write(workdir, val, fname)
        else:
            print(f"Skip {func.__name__}.")
        return val
    return wrapper




def file_result(filename):
    """
    Reuse the file result.
    :param func:
    :return:
    """
    def decorator(func):
        def wrapper(workdir, *args, **kwargs):
            fname = workdir / filename
            if not fname.exists():
                print(f"Execute {filename} = {func.__name__}  ...", end='')
                start = time.process_time_ns()
                val = func(workdir, filename, *args, **kwargs)
                sec = (time.process_time_ns() - start) / 1e9
                print(f"[{sec}] s.")
            else:
                val = filename
                print(f"Skip {filename} = {func.__name__}.")
            return val

        return wrapper
    return decorator