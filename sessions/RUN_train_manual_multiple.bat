python train_manual.py -path_training_data "E:\full32_redist\1.npy" "E:\full32_redist\2.npy" "E:\full32_redist\3.npy"^
 -path_validation_data "E:\validate32_redist\1.npy" "E:\validate32_redist\2.npy" "E:\validate32_redist\3.npy"^
 -validation_frequency=1 -weight_decay_l2=0.0001 -epochs=20 -aug_noise=0.1 -lr_steps 5 10 -resnet_block_sizes 2 2 2 -resnet_filter_sizes 32 32 32 -resnet_n_channels=10

