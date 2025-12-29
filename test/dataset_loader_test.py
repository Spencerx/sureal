import os
import tempfile
import unittest

from sureal.config import SurealConfig
from sureal.dataset_loader import (
    load_dataset,
    load_json_dataset,
    load_yaml_dataset,
    load_python_dataset,
    validate_dataset,
    DatasetValidationError,
    save_dataset_json,
)


class DatasetLoaderTest(unittest.TestCase):
    """Tests for the unified dataset loader."""

    def test_load_json_dataset(self):
        """Test loading a JSON dataset."""
        filepath = SurealConfig.test_resource_path('test_dataset_os_as_dict.json')
        dataset = load_json_dataset(filepath)
        self.assertEqual(dataset.dataset_name, 'test_dataset_os_as_dict')
        self.assertEqual(len(dataset.ref_videos), 2)
        self.assertEqual(len(dataset.dis_videos), 3)
        self.assertEqual(dataset.ref_score, 5.0)

    def test_load_yaml_dataset(self):
        """Test loading a YAML dataset."""
        filepath = SurealConfig.test_resource_path('test_dataset_os_as_dict.yaml')
        dataset = load_yaml_dataset(filepath)
        self.assertEqual(dataset.dataset_name, 'test_dataset_os_as_dict')
        self.assertEqual(len(dataset.ref_videos), 2)
        self.assertEqual(len(dataset.dis_videos), 3)
        self.assertEqual(dataset.ref_score, 5.0)

    def test_load_python_dataset(self):
        """Test loading a Python dataset."""
        filepath = SurealConfig.test_resource_path('test_dataset_os_as_dict.py')
        dataset = load_python_dataset(filepath)
        self.assertEqual(dataset.dataset_name, 'test_dataset_os_as_dict')
        self.assertEqual(len(dataset.ref_videos), 2)
        self.assertEqual(len(dataset.dis_videos), 3)

    def test_load_dataset_auto_detect_json(self):
        """Test auto-detecting JSON format."""
        filepath = SurealConfig.test_resource_path('test_dataset_os_as_dict.json')
        dataset = load_dataset(filepath)
        self.assertEqual(dataset.dataset_name, 'test_dataset_os_as_dict')

    def test_load_dataset_auto_detect_yaml(self):
        """Test auto-detecting YAML format."""
        filepath = SurealConfig.test_resource_path('test_dataset_os_as_dict.yaml')
        dataset = load_dataset(filepath)
        self.assertEqual(dataset.dataset_name, 'test_dataset_os_as_dict')

    def test_load_dataset_auto_detect_python(self):
        """Test auto-detecting Python format."""
        filepath = SurealConfig.test_resource_path('test_dataset_os_as_dict.py')
        dataset = load_dataset(filepath)
        self.assertEqual(dataset.dataset_name, 'test_dataset_os_as_dict')

    def test_load_dataset_unsupported_format(self):
        """Test error on unsupported format."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b'test')
            filepath = f.name
        try:
            with self.assertRaises(ValueError):
                load_dataset(filepath)
        finally:
            os.unlink(filepath)

    def test_load_dataset_file_not_found(self):
        """Test error on missing file."""
        with self.assertRaises(FileNotFoundError):
            load_dataset('/nonexistent/path/file.json')

    def test_validate_dataset_missing_ref_videos(self):
        """Test validation fails without ref_videos."""
        data = {'dis_videos': []}
        with self.assertRaises(DatasetValidationError):
            validate_dataset(data)

    def test_validate_dataset_missing_dis_videos(self):
        """Test validation fails without dis_videos."""
        data = {'ref_videos': []}
        with self.assertRaises(DatasetValidationError):
            validate_dataset(data)

    def test_validate_dataset_missing_os(self):
        """Test validation fails when dis_video missing 'os'."""
        data = {
            'ref_videos': [{'content_id': 0, 'path': 'test.yuv'}],
            'dis_videos': [{'asset_id': 0, 'content_id': 0, 'path': 'test.yuv'}]
        }
        with self.assertRaises(DatasetValidationError):
            validate_dataset(data)

    def test_save_and_load_json_dataset(self):
        """Test round-trip save and load."""
        # Load original
        filepath = SurealConfig.test_resource_path('test_dataset_os_as_dict.json')
        dataset = load_json_dataset(filepath)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            save_dataset_json(dataset, temp_path)

            # Load saved
            dataset2 = load_json_dataset(temp_path)

            self.assertEqual(dataset.dataset_name, dataset2.dataset_name)
            self.assertEqual(len(dataset.dis_videos), len(dataset2.dis_videos))
        finally:
            os.unlink(temp_path)

    def test_json_and_python_equivalent(self):
        """Test that JSON and Python datasets load equivalently."""
        json_path = SurealConfig.test_resource_path('test_dataset_os_as_dict.json')
        py_path = SurealConfig.test_resource_path('test_dataset_os_as_dict.py')

        json_dataset = load_dataset(json_path)
        py_dataset = load_dataset(py_path)

        self.assertEqual(json_dataset.dataset_name, py_dataset.dataset_name)
        self.assertEqual(json_dataset.ref_score, py_dataset.ref_score)
        self.assertEqual(len(json_dataset.ref_videos), len(py_dataset.ref_videos))
        self.assertEqual(len(json_dataset.dis_videos), len(py_dataset.dis_videos))

    def test_yaml_and_json_equivalent(self):
        """Test that YAML and JSON datasets load equivalently."""
        yaml_path = SurealConfig.test_resource_path('test_dataset_os_as_dict.yaml')
        json_path = SurealConfig.test_resource_path('test_dataset_os_as_dict.json')

        yaml_dataset = load_dataset(yaml_path)
        json_dataset = load_dataset(json_path)

        self.assertEqual(yaml_dataset.dataset_name, json_dataset.dataset_name)
        self.assertEqual(yaml_dataset.ref_score, json_dataset.ref_score)
        self.assertEqual(len(yaml_dataset.ref_videos), len(json_dataset.ref_videos))
        self.assertEqual(len(yaml_dataset.dis_videos), len(json_dataset.dis_videos))


if __name__ == '__main__':
    unittest.main()
