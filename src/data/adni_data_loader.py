# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import os
import numpy as np
import logging
import gc
import h5py
from skimage import transform
import math

import utils

import pandas as pd
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# NC==CN (it's a bug I accidentally introduced
diagnosis_dict = {'CN': 0, 'NC': 0, 'MCI': 1, 'AD': 2}
gender_dict = {'Male': 0, 'Female': 1}
viscode_dict = {'bl': 0, 'm03': 1, 'm06': 2, 'm12': 3, 'm18': 4, 'm24': 5, 'm36': 6, 'm48': 7, 'm60': 8, 'm72': 9,
                'm84': 10, 'm96': 11, 'm108': 12, 'm120': 13}

# Maximum number of data points that can be in memory at any time
MAX_WRITE_BUFFER = 5


def fix_nan_and_unknown(input, target_data_format=lambda x: x, nan_val=-1, unknown_val=-2):
    if math.isnan(float(input)):
        input = nan_val
    elif input == 'unknown':
        input = unknown_val

    return target_data_format(input)


def crop_or_pad_slice_to_size(image, target_size, offset=None):

    if offset is None:
        offset = (0, 0, 0)

    x_t, y_t, z_t = target_size
    x_s, y_s, z_s = image.shape

    output_volume = np.min(image)*np.ones((x_t, y_t, z_t))

    x_d = abs(x_t - x_s) // 2 + offset[0]
    y_d = abs(y_t - y_s) // 2 + offset[1]
    z_d = abs(z_t - z_s) // 2 + offset[2]

    t_ranges = []
    s_ranges = []

    for t, s, d in zip([x_t, y_t, z_t], [x_s, y_s, z_s], [x_d, y_d, z_d]):

        if t < s:
            t_range = slice(t)
            s_range = slice(d, d + t)
        else:
            t_range = slice(d, d + s)
            s_range = slice(s)

        t_ranges.append(t_range)
        s_ranges.append(s_range)

    output_volume[t_ranges[0], t_ranges[1], t_ranges[2]
                  ] = image[s_ranges[0], s_ranges[1], s_ranges[2]]

    return output_volume


def prepare_data(input_folder, output_file, size, target_resolution, labels_list, rescale_to_one, offset=None, image_postfix='.nii.gz'):
    '''
    Main function that prepares a dataset from the raw challenge data to an hdf5 dataset
    '''

    csv_summary_file = os.path.join(input_folder, 'summary_alldata.csv')

    summary = pd.read_csv(csv_summary_file)
    # Use only cases that have imaging data (obs)
    summary = summary.loc[summary['image_exists'] == True]
    # Don't use images with unknown diagnosis
    summary = summary.loc[~(summary['diagnosis_3cat'] == 'unknown')]

    # Get list of unique rids
    rids = summary.rid.unique()

    # Get initial diagnosis for rough stratification
    diagnoses = []
    for rid in rids:
        diagnoses.append(
            summary.loc[summary['rid'] == rid]['diagnosis_3cat'].values[0])

    train_and_val_rids, test_rids, train_and_val_diagnoses, _ = train_test_split(
        rids, diagnoses, test_size=0.2, stratify=diagnoses)
    train_rids, val_rids = train_test_split(
        train_and_val_rids, test_size=0.2, stratify=train_and_val_diagnoses)

    print(len(train_rids), len(test_rids), len(val_rids))

    # n_images_train = len(summary.loc[summary['rid'].isin(train_rids)])
    # n_images_test = len(summary.loc[summary['rid'].isin(test_rids)])
    # n_images_val = len(summary.loc[summary['rid'].isin(val_rids)])

    hdf5_file = h5py.File(output_file, "w")

    diag_list = {'test': [], 'train': [], 'val': []}
    weight_list = {'test': [], 'train': [], 'val': []}
    age_list = {'test': [], 'train': [], 'val': []}
    gender_list = {'test': [], 'train': [], 'val': []}
    rid_list = {'test': [], 'train': [], 'val': []}
    viscode_list = {'test': [], 'train': [], 'val': []}
    adas13_list = {'test': [], 'train': [], 'val': []}
    mmse_list = {'test': [], 'train': [], 'val': []}
    field_strength_list = {'test': [], 'train': [], 'val': []}

    file_list = {'test': [], 'train': [], 'val': []}

    logging.info('Counting files and parsing meta data...')

    for train_test, set_rids in zip(['train', 'test', 'val'], [train_rids, test_rids, val_rids]):

        for ii, row in summary.iterrows():

            rid = row['rid']
            if rid not in set_rids:
                continue

            diagnosis_str = row['diagnosis_3cat']
            diagnosis = diagnosis_dict[diagnosis_str]
            if diagnosis not in labels_list:
                continue

            rid_list[train_test].append(rid)
            diag_list[train_test].append(diagnosis)

            viscode = row['viscode']
            viscode_list[train_test].append(viscode_dict[viscode])
            weight_list[train_test].append(row['weight'])
            age_list[train_test].append(row['age'])
            gender_list[train_test].append(gender_dict[row['gender']])
            adas13_list[train_test].append(fix_nan_and_unknown(
                row['adas13'], target_data_format=np.float32))
            mmse_list[train_test].append(fix_nan_and_unknown(
                row['mmse'], target_data_format=np.uint8))

            field_strength = row['field_strength']
            field_strength_list[train_test].append(field_strength)

            phase = row['phase']

            file_name = 'rid_%s/%s_%sT_%s_rid%s_%s%s' % (str(rid).zfill(4),
                                                         phase.lower(),
                                                         str(field_strength),
                                                         diagnosis_str,
                                                         str(rid).zfill(4),
                                                         viscode,
                                                         image_postfix)
            file_list[train_test].append(os.path.join(input_folder, file_name))

    # Write the small datasets
    for tt in ['test', 'train', 'val']:

        hdf5_file.create_dataset(
            'rid_%s' % tt, data=np.asarray(rid_list[tt], dtype=np.uint16))
        hdf5_file.create_dataset('viscode_%s' % tt, data=np.asarray(
            viscode_list[tt], dtype=np.uint8))
        hdf5_file.create_dataset(
            'diagnosis_%s' % tt, data=np.asarray(diag_list[tt], dtype=np.uint8))
        hdf5_file.create_dataset('age_%s' % tt, data=np.asarray(
            age_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('weight_%s' % tt, data=np.asarray(
            weight_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('gender_%s' % tt, data=np.asarray(
            gender_list[tt], dtype=np.uint8))
        hdf5_file.create_dataset('adas13_%s' % tt, data=np.asarray(
            adas13_list[tt], dtype=np.float32))
        hdf5_file.create_dataset(
            'mmse_%s' % tt, data=np.asarray(mmse_list[tt], dtype=np.uint8))
        hdf5_file.create_dataset('field_strength_%s' % tt, data=np.asarray(
            field_strength_list[tt], dtype=np.float16))

    n_train = len(file_list['train'])
    n_test = len(file_list['test'])
    n_val = len(file_list['val'])

    # assert n_train == n_images_train, 'Mismatch in data sizes, %d not == %d' % (n_train, n_images_train)
    # assert n_test == n_images_test, 'Mismatch in data sizes, %d not == %d' % (n_test, n_images_test)
    # assert n_val == n_images_val, 'Mismatch in data sizes, %d not == %d' % (n_val, n_images_val)

    # Create datasets for images and masks
    data = {}
    for tt, num_points in zip(['test', 'train', 'val'], [n_test, n_train, n_val]):
        data['images_%s' % tt] = hdf5_file.create_dataset(
            "images_%s" % tt, [num_points] + list(size), dtype=np.float32)

    img_list = {'test': [], 'train': [], 'val': []}

    logging.info('Parsing image files')

    for train_test in ['test', 'train', 'val']:

        write_buffer = 0
        counter_from = 0

        for file in file_list[train_test]:

            logging.info(
                '-----------------------------------------------------------')
            logging.info('Doing: %s' % file)

            img_dat = utils.load_nii(file)
            img = img_dat[0].copy()

            pixel_size = (img_dat[2].structarr['pixdim'][1],
                          img_dat[2].structarr['pixdim'][2],
                          img_dat[2].structarr['pixdim'][3])

            logging.info('Pixel size:')
            logging.info(pixel_size)

            scale_vector = [pixel_size[0] / target_resolution[0],
                            pixel_size[1] / target_resolution[1],
                            pixel_size[2] / target_resolution[2]]

            img_scaled = transform.rescale(img,
                                           scale_vector,
                                           order=1,
                                           preserve_range=True,
                                           multichannel=False,
                                           mode='constant')

            img_resized = crop_or_pad_slice_to_size(
                img_scaled, size, offset=offset)

            if rescale_to_one:
                img_resized = utils.map_image_to_intensity_range(
                    img_resized, -1, 1, percentiles=5)
            else:
                img_resized = utils.normalise_image(img_resized)

            ### DEBUGGING ############################################
            # utils.create_and_save_nii(img_resized, 'debug.nii.gz')
            # exit()
            #########################################################

            img_list[train_test].append(img_resized)

            write_buffer += 1

            if write_buffer >= MAX_WRITE_BUFFER:

                counter_to = counter_from + write_buffer
                _write_range_to_hdf5(
                    data, train_test, img_list, counter_from, counter_to)
                _release_tmp_memory(img_list, train_test)

                # reset stuff for next iteration
                counter_from = counter_to
                write_buffer = 0

        # after file loop: Write the remaining data

        logging.info('Writing remaining data')
        counter_to = counter_from + write_buffer

        _write_range_to_hdf5(data, train_test, img_list,
                             counter_from, counter_to)
        _release_tmp_memory(img_list, train_test)

    # After test train loop:
    hdf5_file.close()


def _write_range_to_hdf5(hdf5_data, train_test, img_list, counter_from, counter_to):
    '''
    Helper function to write a range of data to the hdf5 datasets
    '''

    logging.info('Writing data from %d to %d' % (counter_from, counter_to))
    img_arr = np.asarray(img_list[train_test], dtype=np.float32)
    hdf5_data['images_%s' % train_test][counter_from:counter_to, ...] = img_arr


def _release_tmp_memory(img_list, train_test):
    '''
    Helper function to reset the tmp lists and free the memory
    '''

    img_list[train_test].clear()
    gc.collect()


def load_and_maybe_process_data(input_folder,
                                preprocessing_folder,
                                size,
                                target_resolution,
                                label_list,
                                offset=None,
                                rescale_to_one=False,
                                force_overwrite=False):
    '''
    This function is used to load and if necessary preprocesses the ACDC challenge data

    :param input_folder: Folder where the raw ACDC challenge data is located
    :param preprocessing_folder: Folder where the proprocessed data should be written to
    :param size: Size of the output slices/volumes in pixels/voxels
    :param target_resolution: Resolution to which the data should resampled. Should have same shape as size
    :param force_overwrite: Set this to True if you want to overwrite already preprocessed data [default: False]

    :return: Returns an h5py.File handle to the dataset
    '''

    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in target_resolution])

    lbl_str = '_'.join([str(i) for i in label_list])

    if rescale_to_one:
        rescale_postfix = '_intrangeone'
    else:
        rescale_postfix = ''

    if offset is not None:
        offset_postfix = '_offset_%d_%d_%d' % offset
    else:
        offset_postfix = ''

    data_file_name = 'all_data_size_%s_res_%s_lbl_%s%s%s.hdf5' % (
        size_str, res_str, lbl_str, rescale_postfix, offset_postfix)
    data_file_path = os.path.join(preprocessing_folder, data_file_name)

    utils.makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path) or force_overwrite:
        logging.info(
            'This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(input_folder, data_file_path, size, target_resolution,
                     label_list, offset=offset, rescale_to_one=rescale_to_one)
    else:
        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')


if __name__ == '__main__':

    # input_folder = '/itet-stor/baumgach/bmicdatasets_bmicnas01/Processed/ADNI_Christian/ADNI_allfixed_no_skullstrip'
    # preprocessing_folder = 'preproc_data/allfixed_noskull'

    # input_folder = '/itet-stor/baumgach/bmicdatasets_bmicnas01/Processed/ADNI_Christian/ADNI_allfixed_allPP_robex'
    # preprocessing_folder = 'preproc_data/allfixed_noskull'

    # d=load_and_maybe_process_data(input_folder, preprocessing_folder, (146, 192, 125), (1.36, 1.36, 1.0), force_overwrite=True)
    # d=load_and_maybe_process_data(input_folder, preprocessing_folder, (130, 160, 113), (1.5, 1.5, 1.5), (0,2), force_overwrite=True)
    # d=load_and_maybe_process_data(input_folder, preprocessing_folder, (128, 160, 112), (1.5, 1.5, 1.5), (0,2), force_overwrite=False, rescale_to_one=True)
    # d=load_and_maybe_process_data(input_folder, preprocessing_folder, (128, 160, 112), (1.5, 1.5, 1.5), (0,2), force_overwrite=False, rescale_to_one=True)
    # d=load_and_maybe_process_data(input_folder, preprocessing_folder, (64, 80, 64), (1.5, 1.5, 1.5), (0,2), offset=(0,0,-10), force_overwrite=False, rescale_to_one=True)

    import config.system as sys_config

    data_root = os.path.join(sys_config.data_root, 'ADNI_allfixed_allPP_robex')
    preproc_folder = os.path.join(
        sys_config.project_root, 'data/preproc_data/allfixed_noskull')

    d = load_and_maybe_process_data(data_root, preproc_folder, (128, 160, 112), (
        1.3, 1.3, 1.3), (0, 1, 2), offset=None, force_overwrite=True, rescale_to_one=False)
