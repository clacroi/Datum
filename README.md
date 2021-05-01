# Datum : manage your datasets on disk and in memory for Machine Learning and Deep Learning

### Description

*Datum* is a datasets management library for different tasks:

* loading datasets on disk in memory.

* manipulating constructed datasets :
  * querying/modifying/deleting elements
  * filtering the dataset
  * iterating over the dataset

* formatting the dataset at a specific format

Datum also provides a set of utility functions for manipulating standard Python data structures and stored files.

### Installation

Datum and its dependencies can be installed directly with pip3 as it is structured as a Python package. Don't forget to use -e option to incorporate any Datum code modification without desinstalling/reinstalling the package.

<pre>
pip3 install -e <path_to_Datum>
</pre>

### Currently supported datasets:

|Dataset|Download link|
|---|---|---|
|VOC (detection)|http://host.robots.ox.ac.uk/pascal/VOC/|
|COCO (Detection)|http://cocodataset.org/|
|Standard Segmenation datasets|N/A|

Adding a dataset is simple: the `datum.readers.DatasetReader`, which is an abstract class, needs to be implemented for feeding a Dataset class from your target database format.

### Examples

For use examples of Datum library, you can check :

* datum/notebooks/demonstrator_basic_features.ipynb.ipynb (notebook)
* datum/notebooks/demonstrator_advanced_features.ipynb (notebook)
* datum/notebooks/benchmarks.ipynb (notebook)
