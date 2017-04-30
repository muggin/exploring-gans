import os
import urllib
import argparse
import shutil
import tarfile
import tempfile
import numpy as np
import cPickle as pickle
import scipy.ndimage as simage


def download_and_extract_data(data_urls, extract_path):
    """ Download data and extract from archive if applicable """
    for data_url in data_urls:
        download_path = os.path.join(extract_path, os.path.basename(data_url))
        data_file, _ = urllib.urlretrieve(data_url, download_path)

        if 'tar' in data_file or 'gz' in data_file:
            with tarfile.open(data_file, "r:gz") as tar:
                tar.extractall(extract_path)
            os.remove(data_file)


def repack_cifar(data_path):
    """ Repack cifar data from pickled batches format to numpy arrays. """
    archive_contents = []

    for _, _, files in os.walk(data_path):
        for filename in files:
            if 'data' in filename or 'test' in filename:
                file_path = os.path.join(data_path, filename)

                with open(file_path, 'rb') as fd:
                    # get and reshape data
                    file_content = pickle.load(fd)['data']
                    archive_contents.append(file_content.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))

    # combine data into single numpy array
    return np.vstack(archive_contents).astype(np.uint8)


def repack_mnist(data_path):
    """ Repack mnist data from csv format to numpy arrays. """
    archive_contents = []

    for _, _, files in os.walk(data_path):
        for filename in files:
            if 'csv' in filename:
                file_path = os.path.join(data_path, filename)

                with open(file_path, 'r') as fd:
                    for line in fd:
                        image = np.array([int(x) for x in line.split(',')[1:]])
                        archive_contents.append(image.reshape(28, 28, 1))
    return np.array(archive_contents).astype(np.uint8)


def repack_lfw(data_path):
    """ Repack lfw data from dir hierarchy to numpy arrays """
    archive_contents = []

    for root, _, files in os.walk(data_path):
        for filename in files:
            if '.jpg' in filename:
                file_path = os.path.join(root, filename)
                file_content = simage.imread(file_path)
                archive_contents.append(file_content[np.newaxis, :])

    # combine data into single numpy array
    return np.vstack(archive_contents).astype(np.uint8)


def dump_data_to_disk(data, data_path):
    """ Serialize numpy arrays using npy """
    with open(data_path, 'wb') as fd:
        np.save(fd, data)


datasets_meta = {
    'cifar': {'urls': ['https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'],
              'archive_name': 'cifar-10-batches-py',
              'data_handler': repack_cifar},

    'lfw':   {'urls': ['http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz'],
              'archive_name': 'lfw-deepfunneled',
              'data_handler': repack_lfw},

    'mnist': {'urls': ['https://pjreddie.com/media/files/mnist_train.csv',
                       'https://pjreddie.com/media/files/mnist_test.csv'],
              'archive_name': '',
              'data_handler': repack_mnist}
}

if __name__ == '__main__':
    # parse args
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', required=True, help='Dataset name', choices=datasets_meta.keys())
    ap.add_argument('-p', '--output', required=False, help='Path to output file', default='./')
    args = vars(ap.parse_args())
    dataset_name = args['dataset']
    output_path = args['output']

    # make temp dir
    temp_dir = tempfile.mkdtemp(dir='./')

    # find dataset metadata
    data_meta = datasets_meta[dataset_name]

    try:
        # download archive
        print 'Downloading data archive...'
        download_and_extract_data(data_meta['urls'], temp_dir)
        data_path = os.path.join(temp_dir, data_meta['archive_name'])

        # repack data
        print 'Repacking data to numpy format...'
        data_repacker = data_meta['data_handler']
        data_prepared = data_repacker(data_path)
        print 'Prepared data shape:', data_prepared.shape

        # pickle data
        print 'Saving data arrays to disk...'
        output_file = os.path.join(output_path, dataset_name + '.npy')
        dump_data_to_disk(data_prepared, output_file)

    finally:
        # cleanup extracted data
        shutil.rmtree(temp_dir)
