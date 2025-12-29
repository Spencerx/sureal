"""
Unified dataset loader for SUREAL.

Supports loading datasets from:
- JSON files (.json) - recommended default format
- YAML files (.yaml, .yml) - human-readable alternative
- Python files (.py) - legacy format, still supported

Example JSON dataset structure:
{
    "dataset_name": "my_dataset",
    "ref_score": 5.0,
    "ref_videos": [
        {"content_id": 0, "content_name": "video1", "path": "ref/video1.yuv"}
    ],
    "dis_videos": [
        {"asset_id": 0, "content_id": 0, "path": "dis/video1_q1.yuv",
         "os": {"subject1": 4, "subject2": 5}}
    ]
}
"""

import json
import os
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"


class DatasetValidationError(Exception):
    """Raised when dataset validation fails."""
    pass


def validate_dataset(data: Dict[str, Any], filepath: Optional[str] = None) -> None:
    """
    Validate that a dataset dictionary has the required structure.

    Required fields:
    - ref_videos: list of dicts with 'content_id', 'content_name', 'path'
    - dis_videos: list of dicts with 'asset_id', 'content_id', 'path', 'os'

    Optional but common fields:
    - dataset_name: str
    - ref_score: float (default reference score)
    - yuv_fmt, width, height: video format info

    Raises:
        DatasetValidationError: If validation fails
    """
    source = f" in {filepath}" if filepath else ""

    # Check required fields
    if 'ref_videos' not in data:
        raise DatasetValidationError(f"Missing required field 'ref_videos'{source}")
    if 'dis_videos' not in data:
        raise DatasetValidationError(f"Missing required field 'dis_videos'{source}")

    # Validate ref_videos
    if not isinstance(data['ref_videos'], list):
        raise DatasetValidationError(f"'ref_videos' must be a list{source}")

    for i, ref_video in enumerate(data['ref_videos']):
        if not isinstance(ref_video, dict):
            raise DatasetValidationError(
                f"ref_videos[{i}] must be a dict{source}")
        for field in ['content_id', 'path']:
            if field not in ref_video:
                raise DatasetValidationError(
                    f"ref_videos[{i}] missing required field '{field}'{source}")

    # Validate dis_videos
    if not isinstance(data['dis_videos'], list):
        raise DatasetValidationError(f"'dis_videos' must be a list{source}")

    for i, dis_video in enumerate(data['dis_videos']):
        if not isinstance(dis_video, dict):
            raise DatasetValidationError(
                f"dis_videos[{i}] must be a dict{source}")
        for field in ['asset_id', 'content_id', 'path', 'os']:
            if field not in dis_video:
                raise DatasetValidationError(
                    f"dis_videos[{i}] missing required field '{field}'{source}")

        # Validate 'os' field structure
        os_field = dis_video['os']
        if not isinstance(os_field, (list, tuple, dict)):
            raise DatasetValidationError(
                f"dis_videos[{i}]['os'] must be a list, tuple, or dict{source}")


def _dict_to_namespace(data: Dict[str, Any]) -> Namespace:
    """Convert a dictionary to an argparse.Namespace object."""
    return Namespace(**data)


def load_json_dataset(filepath: str) -> Namespace:
    """
    Load a dataset from a JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Namespace object containing the dataset
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    validate_dataset(data, filepath)
    return _dict_to_namespace(data)


def load_yaml_dataset(filepath: str) -> Namespace:
    """
    Load a dataset from a YAML file.

    Args:
        filepath: Path to the YAML file

    Returns:
        Namespace object containing the dataset

    Raises:
        ImportError: If PyYAML is not installed
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required to load YAML datasets. "
            "Install it with: pip install pyyaml"
        )

    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    validate_dataset(data, filepath)
    return _dict_to_namespace(data)


def load_python_dataset(filepath: str) -> Any:
    """
    Load a dataset from a Python file (legacy format).

    The Python file should define module-level variables like:
    - dataset_name
    - ref_videos
    - dis_videos
    - ref_score
    etc.

    Args:
        filepath: Path to the Python file

    Returns:
        Module object containing the dataset variables
    """
    from sureal.tools.misc import get_file_name_without_extension

    filename = get_file_name_without_extension(filepath)
    try:
        from importlib.machinery import SourceFileLoader
        ret = SourceFileLoader(filename, filepath).load_module()
    except ImportError:
        import imp
        ret = imp.load_source(filename, filepath)
    return ret


def load_dataset(filepath: str) -> Union[Namespace, Any]:
    """
    Load a dataset from any supported format.

    The format is auto-detected based on file extension:
    - .json: JSON format (recommended)
    - .yaml, .yml: YAML format
    - .py: Python module format (legacy)

    Args:
        filepath: Path to the dataset file

    Returns:
        Dataset object (Namespace for JSON/YAML, module for Python)

    Raises:
        ValueError: If the file format is not supported
        FileNotFoundError: If the file does not exist
        DatasetValidationError: If the dataset structure is invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    ext = Path(filepath).suffix.lower()

    loaders = {
        '.json': load_json_dataset,
        '.yaml': load_yaml_dataset,
        '.yml': load_yaml_dataset,
        '.py': load_python_dataset,
    }

    if ext not in loaders:
        supported = ', '.join(loaders.keys())
        raise ValueError(
            f"Unsupported dataset format: {ext}. Supported formats: {supported}"
        )

    return loaders[ext](filepath)


def dataset_to_dict(dataset: Union[Namespace, Any]) -> Dict[str, Any]:
    """
    Convert a dataset (Namespace or module) to a dictionary.

    Args:
        dataset: Dataset object (Namespace or Python module)

    Returns:
        Dictionary representation of the dataset
    """
    if isinstance(dataset, Namespace):
        return vars(dataset)
    else:
        # Python module - extract relevant attributes
        return {
            key: getattr(dataset, key)
            for key in dir(dataset)
            if not key.startswith('_')
        }


def save_dataset_json(dataset: Union[Namespace, Any, Dict], filepath: str,
                      indent: int = 2) -> None:
    """
    Save a dataset to a JSON file.

    Args:
        dataset: Dataset object (Namespace, module, or dict)
        filepath: Output file path
        indent: JSON indentation (default 2)
    """
    if isinstance(dataset, dict):
        data = dataset
    else:
        data = dataset_to_dict(dataset)

    # Filter out non-serializable items and internal attributes
    serializable_data = {}
    for key, value in data.items():
        if key.startswith('_'):
            continue
        try:
            json.dumps(value)
            serializable_data[key] = value
        except (TypeError, ValueError):
            # Skip non-serializable values (like module references)
            pass

    with open(filepath, 'w') as f:
        json.dump(serializable_data, f, indent=indent)
