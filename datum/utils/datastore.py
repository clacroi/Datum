"""
Datum "Datastore" features utility functions.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import csv

from datum.datasets import Dataset
from datum.readers import load_datum_dataset
from datum.formatters import DatumFormatter


DATUM_ROOT = Path(__file__).resolve().parents[1]
DATUM_DATASTORES = DATUM_ROOT / 'datastores'
DATUM_DATASTORES.mkdir(exist_ok=True)


def load_datastore_csv(path: Path):
    with open(path, newline='\n') as csvfile:
        datasets = []
        reader = csv.reader(csvfile, delimiter=',')
        _ = next(reader)
        for row in reader:
            if len(row) != 2:
                raise ValueError('Datastore CSV lines should have length 2 ({})'.format(row))
            datasets.append((row[0], Path(row[1])))
    return datasets

def list_existing_datastores():
    print('# DATASTORES')
    datastores = [x for x in DATUM_DATASTORES.glob('*.csv')]
    for datastore in datastores:
        print('## {}'.format(datastore.stem))


# TODO : it should be checked that each dataset is indeed a Datum dataset
def list_existing_datasets(datastore: Optional[str] = None):
    datastores = [x for x in DATUM_DATASTORES.glob('*.csv')]
    if datastore:
        if DATUM_DATASTORES / (datastore + '.csv') not in datastores:
            raise ValueError('Datastore {}.csv does not exist'.format(datastore))
        else:
            datasets = load_datastore_csv(DATUM_DATASTORES / (datastore + '.csv'))
        print('DATASTORE {} ({})'.format(datastore, DATUM_DATASTORES / (datastore + '.csv')))
        for dataset_name, dataset_dir in datasets:
            print('{}.{} (path : {})'.format(datastore, dataset_name, str(dataset_dir)))
        print('\n')

    else:
        for datastore in datastores:
            datasets = load_datastore_csv(datastore)
            print('\n# DATASTORE {} ({})'.format(datastore.stem, datastore))
            for dataset_name, dataset_dir in datasets:
                print('## {} (path : {})'.format(dataset_name, str(dataset_dir)))
        print('\n')


def register_dataset(dataset_root_dir: Path,
                     dataset_name: str,
                     datastore: str):
    dataset_root_dir = Path(dataset_root_dir)
    # TODO : check that there is indeed a Datum dataset @dataset_root_dir

    # load datastore if exists or construct new one
    datastores = [x for x in DATUM_DATASTORES.glob('*.csv')]
    datastore_path = DATUM_DATASTORES / (datastore + '.csv')
    if datastore_path not in datastores:
        # TODO : create the datastore if does not exist
        with open(datastore_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['dataset_name', 'dataset_root_dir'])
    else:
        # verify that no dataset with same name / root_dir already exist
        datasets = load_datastore_csv(DATUM_DATASTORES / (datastore + '.csv'))
        for existing_dataset_name, existing_dataset_root_dir in datasets:
            if dataset_name == existing_dataset_name:
                raise ValueError('Dataset with name {} already exists in datastore {}'
                                .format(existing_dataset_name, datastore))
            if dataset_root_dir == existing_dataset_root_dir:
                raise ValueError('Dataset with root_dir {} already exists in datastore {}'
                                .format(existing_dataset_root_dir, datastore))

    # Update datastore
    with open(datastore_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([dataset_name, dataset_root_dir])


def load_dataset(name, datastore: Optional[str] = None):
    datastores = [x for x in DATUM_DATASTORES.glob('*.csv')]
    if not datastore and '.' in name:
        datastore, name = name.split('.')
    if datastore:
        if DATUM_DATASTORES / (datastore + '.csv') not in datastores:
            raise ValueError('Datastore {}.csv does not exist'.format(datastore))
        datasets = load_datastore_csv(DATUM_DATASTORES / (datastore + '.csv'))
        # TODO : replace by less lines
        for dataset_name, dataset_root_dir in datasets:
            if dataset_name == name:
                found = True
                break
        if not found:
            raise ValueError('Dataset with name {} does not exist in datastore {}'
                             .format(name, datastore))
        return load_datum_dataset(dataset_root_dir)

    for datastore in datastores:
        datasets = load_datastore_csv(datastore)
        # TODO : replace by less lines
        found = False
        for dataset_name, dataset_root_dir in datasets:
            if dataset_name == name:
                found = True
                break
        if not found:
            raise ValueError('Dataset with name {} does not exist in datastore {}'
                             .format(name, datastore))
        return load_datum_dataset(dataset_root_dir)
