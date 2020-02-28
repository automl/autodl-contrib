import os
import shutil
import sys
from sys import path

import numpy as np
import tensorflow as tf
from PIL import Image

from check_n_format import format_automatically
from icgen import ICGen
from icgen.datasets.names import DATASETS_SMALL

ICGEN_DIR = "/data/aad/image_datasets/icgen/icgen"
path.append(ICGEN_DIR)

tf.compat.v1.enable_eager_execution()


def dataset_exists(dataset, dataset_dir):
    if os.path.isdir(os.path.join(dataset_dir, dataset)):
        return True
    else:
        return False


def convert_to_images(dataset, dataset_dir, dataset_name, split=None):
    split_dir = os.path.join(dataset_dir, dataset_name + '/' + split)
    if not os.path.isdir(split_dir):
        os.mkdir(split_dir)

    label_count_dict = {}

    for data, label in dataset:
        # store correct label
        if label in label_count_dict:
            label_count_dict[label] += 1
        else:
            label_count_dict[label] = 0

        file_name = os.path.join(
            split_dir, split + '-' + str(label) + '-' +
            str(label_count_dict[label]) + '.png')

        # store images/videos
        if len(data.shape) == 3:  # image
            if data.shape[2] == 1:  # grayscale
                color_mode = 'L'
                data = np.squeeze(data)
            elif data.shape[2] == 3:  # RGB
                color_mode = 'RGB'
            else:
                raise ValueError('unknown number of channels: ' +
                                 str(data.shape[2]))

            image = Image.fromarray(data, color_mode)
            image.save(file_name, compress_level=0)

    # write private.info
    info_text = "title : '" + str(dataset_name) +"' \n" \
                 + "name : '" + str(dataset_name) +"' \n" \
                 + "keywords : '' \n" \
                 + "authors : '' \n" \
                 + "resource_url : '' \n" \
                 + "contact_name : '' \n" \
                 + "contact_url : '' \n" \
                 + "license : '' \n" \
                 + "date_created : '' \n" \
                 + "past_usage : '' \n" \
                 + "description : '' \n" \
                 + "preparation : '' \n" \
                 + "representation : 'pixels' \n" \
                 + "remarks : ''"

    info_file = os.path.join(split_dir, 'private.info')
    with open(info_file, 'w') as f:
        f.write(info_text)

    # write labels.csv
    labels_file = os.path.join(split_dir, 'labels.csv')
    with open(labels_file, 'w') as f:
        f.write('FileName,Labels\n')
        for label, count in label_count_dict.items():
            for i in range(int(count) + 1):
                label_text = split + '-' + str(label) + '-' + str(
                    i) + '.png' + ',' + str(label) + '\n'
                f.write(label_text)


def convert_to_autodl(dataset_name, dataset_dir, goal_dir):
    train_dir = os.path.join(dataset_dir, dataset_name + '/train')
    test_dir = os.path.join(dataset_dir, dataset_name + '/test')

    if not os.path.isdir(train_dir):
        print('Conversion to AutoDL dataset failed')

    file_name = os.path.join(train_dir, 'train-0-0.png')
    image = np.array(Image.open(file_name))

    if len(image.shape) == 2:
        num_channels = 1
    else:
        num_channels = image.shape[2]

    print('------- automatically format train ------- ' + dataset_name)
    format_automatically(train_dir, 1.0, num_channels)
    print('------- automatically format test ------- ' + dataset_name)
    format_automatically(test_dir, 0.0, num_channels)
    print('------- merge train test ------- ' + dataset_name)
    merge_train_test_folders(dataset_name, dataset_dir, goal_dir)
    print('------- fix num samples ------- ' + dataset_name)
    fix_num_samples(dataset_name, goal_dir)


def merge_train_test_folders(dataset, dataset_dir, goal_dir):
    train_dir = os.path.join(dataset_dir,
                             dataset + '/train_formatted/' + dataset)
    test_dir = os.path.join(dataset_dir,
                            dataset + '/test_formatted/' + dataset)
    merged_dir = os.path.join(goal_dir, dataset)

    if os.path.isdir(merged_dir):
        shutil.rmtree(merged_dir)

    # copy training and test data
    train_subdir = dataset + '.data/train'
    test_subdir = dataset + '.data/test'
    shutil.copytree(os.path.join(test_dir, test_subdir),
                    os.path.join(merged_dir, test_subdir))
    shutil.copytree(os.path.join(train_dir, train_subdir),
                    os.path.join(merged_dir, train_subdir))

    # copy metadata
    for elem in ['public.info', dataset + '.solution']:
        shutil.copy(os.path.join(test_dir, elem),
                    os.path.join(merged_dir, elem))


def fix_num_samples(dataset, goal_dir):
    num_samples_train = get_num_samples_from_textproto_file(
        dataset, goal_dir, True)
    num_samples_test = get_num_samples_from_textproto_file(
        dataset, goal_dir, False)
    write_num_samples_to_info_file(dataset, goal_dir,
                                   num_samples_train + num_samples_test)


def get_num_samples_from_textproto_file(dataset, goal_dir, use_train):
    if use_train:
        textproto_file = os.path.join(
            goal_dir,
            dataset + '/' + dataset + '.data/train/metadata.textproto')
    else:
        textproto_file = os.path.join(
            goal_dir,
            dataset + '/' + dataset + '.data/test/metadata.textproto')

    with open(textproto_file, 'r') as f:
        lines = f.readlines()
        num_samples = int(lines[1].replace('\n', ' ').split(' ')[1])

    return num_samples


def write_num_samples_to_info_file(dataset, goal_dir, num_samples):
    info_file = os.path.join(goal_dir, dataset + '/public.info')

    with open(info_file, 'r') as f:
        lines = f.readlines()

    lines[0] = 'sample_num : ' + str(num_samples) + '\n'
    # print(lines)
    # print(info_file)

    with open(info_file, 'w') as f:
        f.writelines(lines)


if __name__ == "__main__":
    dataset_names = ["caltech101", "cifar10", "emnist/balanced"]

    info_dir = '/data/aad/image_datasets/icgen/icgen/icgen/infos'
    dataset_dir = '/data/aad/image_datasets/icgen/downloaded_datasets'
    goal_dir = '/data/aad/image_datasets/augmented_datasets'

    for dataset_name in dataset_names:
        ic_benchmark = ICGen(
            data_path=dataset_dir,
            min_resolution=16,
            max_resolution=512,
            max_log_res_deviation=
            1,  # Sample only 1 log resolution from the native one
            min_classes=2,
            max_classes=100,
            min_examples_per_class=20,
            max_examples_per_class=100_000,
        )

        task = ic_benchmark.sample_task(dataset=dataset_name,
                                        augment=True,
                                        resize=True)

        convert_to_images(dataset=task.development_data,
                          dataset_dir=dataset_dir,
                          dataset_name=dataset_name,
                          split='train')

        convert_to_images(dataset=task.test_data,
                          dataset_dir=dataset_dir,
                          dataset_name=dataset_name,
                          split='test')

        convert_to_autodl(dataset_name=dataset_name,
                          dataset_dir=dataset_dir,
                          goal_dir=goal_dir)
