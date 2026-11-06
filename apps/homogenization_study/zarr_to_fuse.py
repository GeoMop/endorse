"""Copy a completed homogenization Zarr dataset into a Zarr Fuse store."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import xarray as xr
import yaml
import zarr_fuse as zf
from zarr.errors import GroupNotFoundError


FIELD_GROUP = "tensors"


def normalize_source_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """Map names from the previous root-level store to the tensor schema."""
    legacy_names = {"param_val": "parameter", "parameter": "param_name"}
    if set(legacy_names).issubset(dataset.variables):
        return dataset.rename(legacy_names)
    return dataset


def open_source_dataset(source_store: str | Path) -> xr.Dataset:
    """Open the schema layout, normalizing the previous root-level layout."""
    try:
        return xr.open_zarr(source_store, group=FIELD_GROUP, consolidated=False)
    except GroupNotFoundError:
        dataset = xr.open_zarr(source_store, consolidated=False)
        return normalize_source_dataset(dataset)


def load_destination_schema(
    schema_path: str | Path,
    *,
    store_url: str | None = None,
    endpoint_url: str | None = None,
) -> Any:
    """Load the Zarr Fuse schema and apply optional destination overrides."""
    path = Path(schema_path)
    with path.open("r", encoding="utf-8") as schema_file:
        schema_config = yaml.safe_load(schema_file)
    schema_config = copy.deepcopy(schema_config)
    if store_url is not None:
        schema_config["ATTRS"]["STORE_URL"] = store_url
    if endpoint_url is not None:
        schema_config["ATTRS"]["S3_ENDPOINT_URL"] = endpoint_url
    return zf.schema.deserialize(schema_config, source_description=str(path))


def convert_to_zarr_fuse(
    source_store: str | Path,
    schema_path: str | Path,
    *,
    store_url: str | None = None,
    endpoint_url: str | None = None,
    workdir: str | Path | None = None,
    overwrite: bool = False,
) -> xr.Dataset:
    """Validate and copy the local ``tensors`` dataset to Zarr Fuse."""
    source = open_source_dataset(source_store)
    schema = load_destination_schema(
        schema_path,
        store_url=store_url,
        endpoint_url=endpoint_url,
    )
    options: dict[str, Any] = {}
    if workdir is not None:
        options["WORKDIR"] = Path(workdir)
    if overwrite:
        zf.remove_store(schema, **options)
    root = zf.open_store(schema, **options)
    tensors = root[FIELD_GROUP]
    tensors.update_from_ds(source)
    return tensors.dataset


def main() -> None:
    """Run the standalone local-Zarr to Zarr-Fuse conversion."""
    default_schema = Path(__file__).resolve().parent / "input_data" / "data_schema.yaml"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_store", type=Path, help="Local homogenization Zarr store.")
    parser.add_argument("--schema", type=Path, default=default_schema, help="Zarr Fuse schema YAML.")
    parser.add_argument("--store-url", help="Override ATTRS.STORE_URL from the schema.")
    parser.add_argument("--endpoint-url", help="Override ATTRS.S3_ENDPOINT_URL from the schema.")
    parser.add_argument("--workdir", type=Path, help="Base directory for a relative local STORE_URL.")
    parser.add_argument("--overwrite", action="store_true", help="Remove the destination before conversion.")
    args = parser.parse_args()
    dataset = convert_to_zarr_fuse(
        args.source_store,
        args.schema,
        store_url=args.store_url,
        endpoint_url=args.endpoint_url,
        workdir=args.workdir,
        overwrite=args.overwrite,
    )
    print(dataset)
    print("First 10 barycenters:\n", dataset["el_barycenter"][:10].to_numpy())
    print("First 10 conductivity:\n", dataset["conductivity"][:10].to_numpy())
    print("First 10 gmsh_id:\n", dataset["gmsh_id"][:10].to_numpy())


if __name__ == "__main__":
    main()
