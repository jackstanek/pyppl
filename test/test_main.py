from unittest import mock

import pytest

from pyppl.__main__ import PickleDumper, PickleLoader


# Unit tests for PickleLoader and PickleDumper
class TestPickleLoader:
    """Unit tests for the PickleLoader class."""

    @mock.patch("builtins.open", new_callable=mock.mock_open)
    @mock.patch("pickle.load")
    def test_pickle_loader_loads_data(self, mock_pickle_load, mock_open):
        """
        Tests that PickleLoader correctly opens the file and loads data
        using pickle.load.
        """
        # Define the data that pickle.load should return
        expected_data = {"key": "value", "number": 123}
        mock_pickle_load.return_value = expected_data

        file_path = "dummy/path/to/file.pkl"
        with PickleLoader(file_path) as loader:
            actual = loader.load()

        # Assertions
        # 1. Check if open was called with the correct path and mode
        mock_open.assert_called_once_with(file_path, mode="rb")
        # 2. Check if pickle.load was called with the file handle returned by open
        mock_pickle_load.assert_called_once_with(mock_open())
        # 3. Check if the loader returned the expected data
        assert actual == expected_data


class TestPickleDumper:
    """Unit tests for the PickleDumper class."""

    @mock.patch("builtins.open", new_callable=mock.mock_open)
    @mock.patch("pickle.Pickler")
    def test_pickle_dumper_dumps_data(self, mock_pickler_class, mock_open):
        """
        Tests that PickleDumper correctly opens the file, creates a pickler,
        and calls dump with the provided data.
        """
        # Mock the instance of Pickler and its dump method
        mock_pickler_instance = mock.Mock()
        mock_pickler_class.return_value = mock_pickler_instance

        file_path = "output/data.pkl"
        dumper = PickleDumper(file_path)
        data_to_dump = {"name": "Test", "id": 456}

        # Use the dumper within a 'with' statement as intended
        with dumper as pd:
            pd.dump(data_to_dump)

        # Assertions
        # 1. Check if open was called with the correct path and 'wb' mode
        mock_open.assert_called_once_with(file_path, mode="wb")
        # 2. Check if pickle.Pickler was instantiated with the file handle
        mock_pickler_class.assert_called_once_with(mock_open())
        # 3. Check if the dump method of the pickler instance was called with the data
        mock_pickler_instance.dump.assert_called_once_with(data_to_dump)
        # 4. Check if the file was closed
        mock_open().close.assert_called_once()

    @mock.patch("builtins.open", new_callable=mock.mock_open)
    @mock.patch("pickle.Pickler")
    def test_pickle_dumper_passes_kwargs_to_open(self, mock_pickler_class, mock_open):
        """
        Tests that PickleDumper correctly passes additional kwargs to open
        when creating the file.
        """
        file_path = "another/output.pkl"

        with PickleDumper(file_path) as pd:
            pd.dump("some data")

        # Check if open was called with the correct path and 'wb' mode, and additional kwargs
        mock_open.assert_called_once_with(file_path, mode="wb")
        mock_open().close.assert_called_once()

    @mock.patch("builtins.open", new_callable=mock.mock_open)
    @mock.patch("pickle.Pickler")
    def test_pickle_dumper_context_manager_behavior(
        self, mock_pickler_class, mock_open
    ):
        """
        Tests the context manager behavior, ensuring the file is closed
        even if an error occurs within the 'with' block.
        """
        mock_pickler_instance = mock.Mock()
        mock_pickler_class.return_value = mock_pickler_instance

        file_path = "error/test.pkl"

        with pytest.raises(ValueError):
            with PickleDumper(file_path) as pd:
                pd.dump("data")
                raise ValueError("Simulated error")

        # Ensure open was called and the file was closed, despite the error
        mock_open.assert_called_once_with(file_path, mode="wb")
        mock_open().close.assert_called_once()
