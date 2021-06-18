import pytest
from allennlp.data import Instance

def test_import_NCBIDiseaseDatasetReader():
    """
    pytest: test if we can import NCBIDiseaseDatasetReader
    """
    from wiser.data.dataset_readers import NCBIDiseaseDatasetReader
    NCBIDiseaseDatasetReader()
    assert True

def get_data():
    """
    use NCBIDiseaseDatasetReader to load a dataset
    """
    from wiser.data.dataset_readers import NCBIDiseaseDatasetReader
    reader = NCBIDiseaseDatasetReader()
    path = "test/toy_data/ncbi.txt"  # assuming we run pytest command on the root wiser folder.
    test_data = reader.read(path)
    return test_data

def test_get_data():
    """
    pytest: test if NCBI-Disease data returned from get_data() is not empty
    """
    data = get_data()
    data = list(data)
    assert data

def test_is_data_instance():
    """
    pytest: test if each NCBI-Disease data instanace returned from get_data()
    is an allennlp.data.Instance
    """
    data = get_data()
    data = list(data)
    assert isinstance(data[0], Instance)

def test_is_data_instance_not_empty():
    """
    pytest: test if each NCBI-Disease data instance returned from get_data()
    is not an empty instance
    """
    data = get_data()
    data = list(data)
    assert len(data[0]['tokens'].tokens) and len(data[0]['tags'].labels)