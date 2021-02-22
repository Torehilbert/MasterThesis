python train_manual.py -path_training_data "E:\full32_redist\1.npy" "E:\full32_redist\2.npy" "E:\full32_redist\3.npy"^
 -path_validation_data "E:\validate32_redist\1.npy" "E:\validate32_redist\2.npy" "E:\validate32_redist\3.npy"^
 -epochs=120 -batch_size=64 -lr_start=0.1 -lr_multiplier=0.1 -lr_steps 50 70 90 -momentum=0.9 -weight_decay_l2=0.0001 -weight_decay_l1_stem=0.0001 -validation_frequency=1 -aug_noise=0.10^
 -resnet_n_channels=10 -resnet_block_sizes 3 5 3 -resnet_filter_size 32 64 128