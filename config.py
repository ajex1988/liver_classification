class STPNSingleSize():
    model_path = '/data/models/single_size_0426'
    max_steps = 50000
    resize_hw = 288
    min_slice = 32
    cuda_device_id = '2'
    crop_diff = 32  # resize_hw-crop_hw
    batch_size = 32
    n_classes = 30
    tfrecord_folder = '/data/liver_internal_3d_augment'
    stpl_kernel_size = [[32, 16, 16], [16, 8, 8], [8, 4, 4]]
    stpl_stride = [[32, 16, 16], [16, 8, 8], [8, 4, 4]]

class STPNTwoSizes():
    max_steps = 50000
    num_epochs = 25  # There are steps_per_size*len(param_list) steps per epoch
    cuda_device_id = '3'
    crop_diff = 32  # resize_hw-crop_hw
    every_n_iter = 100
    batch_size = 32
    n_classes = 30
    tfrecord_folder = '/data/liver_internal_3d_augment'
    model_path = '/data/models/two_sizes_0425'
    param_list = [{'min_slice': 32, 'resize_hw': 256, 'stpl_kernel_size': [[1, 7, 7], [2, 14, 14], [4, 28, 28]],
                   'stpl_stride': [[1, 7, 7], [2, 14, 14], [4, 28, 28]],'tfrecord_folder':'/data/liver_internal_3d_augment'},
                  {'min_slice': 32, 'resize_hw': 288, 'stpl_kernel_size': [[1, 7, 7], [2, 14, 14], [4, 28, 28]],
                   'stpl_stride': [[1, 8, 8], [2, 16, 16], [4, 32, 32]],'tfrecord_folder':'data/liver_internal_3d_augment'}
                  ]