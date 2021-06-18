import pytest
from allennlp.data import Instance

def test_import_NCBIDiseaseDatasetReader():
    from wiser.data.dataset_readers import NCBIDiseaseDatasetReader
    NCBIDiseaseDatasetReader()
    assert True

def get_data():
    from wiser.data.dataset_readers import NCBIDiseaseDatasetReader
    reader = NCBIDiseaseDatasetReader()
    path = "test/toy_data/ncbi.txt"  # assuming we run pytest command on the root wiser folder.
    test_data = reader.read(path)
    return test_data

def test_get_data():
    data = get_data()
    data = list(data)
    assert data

def test_is_data_instance():
    data = get_data()
    data = list(data)
    assert isinstance(data[0], Instance)

def test_is_data_instance_not_empty():
    data = get_data()
    data = list(data)
    assert len(data[0]['tokens'].tokens) and len(data[0]['tags'].labels)