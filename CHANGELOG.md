# Changelog

`v1.0.0-alpha1` (branch: `dev-allennlp-2.5.0`)
- Change from using positional arguments to keyword arguments when creating `allennlp.data.tokenizers.Token(...)` 
- updated the installation files "requirements.txt" and "setup.py".
- remove `lazy` parameter in `NCBIDiseaseDatasetReader` and `WeakLabelDatasetReader`.
- added unit tests for the following class/modules
  - NCBIDiseaseDatasetReader
  - wiser.rules
  - wiser.generative
- fix a bug in `wiser/models/crf_tagger.py` with respect to `metric` and `self._f1_metric` functions, where the mask argument should have Boolean values instead of Float values.

TODO: Update CDRChemicalDatasetReader, CDRDiseaseDatasetReader, CDRCombinedDatasetReader, BioASQDatasetReader, LaptopsDatasetReader, SrlReaderIOB1, MediaDatasetReader
