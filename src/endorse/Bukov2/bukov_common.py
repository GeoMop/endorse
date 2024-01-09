from typing import *
import pickle

from endorse import common
import time
import h5py

import PyPDF2
import os
from pathlib import Path
import numpy as np


def direction_vector(y_phi, z_phi):
    y_phi = y_phi / 180 * np.pi
    z_phi = z_phi / 180 * np.pi
    sy, cy = np.sin(y_phi), np.cos(y_phi)
    sz, cz = np.sin(z_phi), np.cos(z_phi)
    return np.array([cy * cz, sy * cz, sz])


def direction_angles(unit_direction):
    z_angle = np.arcsin(unit_direction[2])
    y_angle = np.arcsin(unit_direction[1] / np.cos(z_angle))
    return 180 * y_angle / np.pi, 180 * z_angle / np.pi


def normalize_sensitivities(data, fn):
    data_2d = data.reshape(-1, data.shape[-1])
    param_scale = fn(data_2d, axis=0)
    data_scaled =  data_2d[:, :] / param_scale[None, :]
    return data_scaled.reshape(data.shape)

def soft_lim_pressure(pressure):
    atm_pressure = 1.013 * 1e5 / 9.89 / 1000    # atmospheric pressure in [m] of water
    abs_pressure = pressure + atm_pressure
    # Limit negative pressure
    vapour_pressure = 0.13   # [m] = 1300 Pa
    epsilon = 10
    soft_max = lambda a, b : 0.5 * (a + b + np.sqrt((a-b) ** 2 + epsilon))
    soft_pressure = soft_max(vapour_pressure, abs_pressure)
    return soft_pressure

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
            import img2pdf
            # Convert PNG to PDF
            pdf_path = file_path.with_suffix('.pdf')
            with open(file_path, "rb") as img_file, open(pdf_path, "wb") as pdf_file:
                pdf_file.write(img2pdf.convert(img_file.read()))

            file_path = pdf_path  # Update file_path to the new PDF file
        if file_path.suffix == '.svg':
            import cairosvg
            # Convert SVG to PDF
            pdf_path = file_path.with_suffix('.pdf')
            #try:
            #with open(file_path, "rb") as img_file, open(pdf_path, "wb") as pdf_file:
            cairosvg.svg2pdf(url=str(file_path), write_to=str(pdf_path))
            #except Exception as e:
            #    print(f"SVG -> PDF conversion failed:\n{e}")
            file_path = pdf_path
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
            print(f"Execute {func.__name__}  ")
            start = time.process_time_ns()
            val = func(workdir, *args, **kwargs)
            sec = (time.process_time_ns() - start) / 1e9
            print(f"... [{sec}] s.")

            pkl_write(workdir, val, fname)
        else:
            print(f"Skip {func.__name__} .")
        return val
    return wrapper




def file_result(filename):
    """
    Reuse the file result.
    :param func:
    :return:
    """
    def decorator(func):
        def wrapper(workdir: Path, *args, **kwargs):
            fname = workdir / filename
            if not fname.exists():
                tmp_fname = fname.with_stem(fname.stem + "_tmp_")
                print(f"Execute {filename} = {func.__name__}  ...", end='')
                start = time.process_time_ns()
                try:
                    val = func(workdir, tmp_fname, *args, **kwargs)
                    assert val == tmp_fname
                    tmp_fname.rename(fname)
                    val = fname
                except Exception as e:
                    tmp_fname.unlink(missing_ok=True)
                    raise e
                sec = (time.process_time_ns() - start) / 1e9
                print(f"[{sec}] s.")
            else:
                val = filename
                print(f"Skip {filename} = {func.__name__}.")
            return val

        return wrapper
    return decorator