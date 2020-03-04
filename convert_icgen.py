import argparse
import os
import shutil
import sys
from sys import path

import numpy as np

import tensorflow as tf
from check_n_format import format_automatically
from icgen import ICGen
from icgen.datasets.names import DATASETS
from more_itertools import flatten
from PIL import Image

ICGEN_DIR = "/data/aad/image_datasets/icgen/icgen"
path.append(ICGEN_DIR)


tf.compat.v1.enable_eager_execution()


def dataset_exists(dataset, dataset_dir):
    if os.path.isdir(os.path.join(dataset_dir, dataset)):
        return True
    else:
        return False


def convert_to_images(dataset,
                      dataset_dir,
                      dataset_name,
                      sub_dir=None,
                      split=None):
    if sub_dir is not None:
        split_dir = os.path.join(dataset_dir, dataset_name, sub_dir, split)
    else:
        split_dir = os.path.join(dataset_dir, dataset_name, split)
    if not os.path.isdir(split_dir):
        os.makedirs(split_dir)

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

    if sub_dir is not None:
        dataset_name = sub_dir

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
    train_dir = os.path.join(dataset_dir, dataset_name, 'train')
    test_dir = os.path.join(dataset_dir, dataset_name, 'test')

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
    merged_dir = os.path.join(
        goal_dir, dataset)  # todo: perhaps specify sub_dir here as well

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
    p = argparse.ArgumentParser(
        "",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--dataset_dir",
                   type=str,
                   default="/data/aad/image_datasets/icgen/datasets")
    p.add_argument("--goal_dir",
                   type=str,
                   default="/data/aad/image_datasets/augmented_datasets")
    p.add_argument("--task_id", default=None)

    args = p.parse_args()

    ic_generator = ICGen(
        data_path=args.dataset_dir,
        min_resolution=16,
        max_resolution=512,
        max_log_res_deviation=
        1,  # Sample only 1 log resolution from the native one
        min_classes=2,
        max_classes=100,
        min_examples_per_class=20,
        max_examples_per_class=100_000,
    )

    datasets_to_exclude = [
        "caltech_birds2010", "binary_alpha_digits", "caltech101"
    ]  #  already done
    datasets_list = [x for x in DATASETS if x not in datasets_to_exclude]
    n_augmented_per_dataset = 49

    if args.task_id is not None:
        index = int(args.task_id) - 1
        datasets_list = [datasets_list[index]]

    for dataset in datasets_list:
        original = ic_generator.sample_task(dataset=dataset,
                                            augment=False,
                                            resize=True)
        dataset_identifier = original.identifier["dataset"] + "_" + "_".join(
            list(map(str, flatten(original.representation.items()))))
        dataset_identifier += "_original"

        convert_to_images(dataset=original.development_data,
                          dataset_dir=args.dataset_dir,
                          dataset_name=dataset,
                          sub_dir=dataset_identifier,
                          split='train')

        convert_to_images(dataset=original.test_data,
                          dataset_dir=args.dataset_dir,
                          dataset_name=dataset,
                          sub_dir=dataset_identifier,
                          split='test')

        convert_to_autodl(dataset_name=dataset_identifier,
                          dataset_dir=os.path.join(args.dataset_dir, dataset),
                          goal_dir=args.goal_dir)

        for _ in range(n_augmented_per_dataset):
            augmented = ic_generator.sample_task(dataset=dataset,
                                                 augment=True,
                                                 resize=True)
            dataset_identifier = augmented.identifier[
                "dataset"] + "_" + "_".join(
                    list(map(str, flatten(augmented.representation.items()))))

            convert_to_images(dataset=original.development_data,
                              dataset_dir=args.dataset_dir,
                              dataset_name=dataset,
                              sub_dir=dataset_identifier,
                              split='train')

            convert_to_images(dataset=original.test_data,
                              dataset_dir=args.dataset_dir,
                              dataset_name=dataset,
                              sub_dir=dataset_identifier,
                              split='test')

            convert_to_autodl(dataset_name=dataset_identifier,
                              dataset_dir=os.path.join(args.dataset_dir,
                                                       dataset),
                              goal_dir=args.goal_dir)
