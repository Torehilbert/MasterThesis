python train_teacher.py -path_training_data "E:\full32_redist\1.npy" "E:\full32_redist\2.npy" "E:\full32_redist\3.npy"^
 -path_validation_data "E:\validate32_redist\1.npy" "E:\validate32_redist\2.npy" "E:\validate32_redist\3.npy"^
 -epochs=100 -lr_steps 30 50 70 -validation_frequency=1 -aug_noise=0.25 -weight_decay_l2=0.0001^
 -resnet_n_channels=10 -resnet_block_sizes 3 5 3 -resnet_filter_sizes 32 64 16 -save_model_epochs 10 30 50 70 85 90 95
