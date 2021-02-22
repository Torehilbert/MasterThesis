python train_manual.py -path_training_data "E:\full32_redist\1.npy" "E:\full32_redist\2.npy" "E:\full32_redist\3.npy"^
 -path_validation_data "E:\validate32_redist\1.npy" "E:\validate32_redist\2.npy" "E:\validate32_redist\3.npy"^
 -epochs=120 -lr_steps 50 70 90 -validation_frequency=1 -aug_noise=0.25 -weight_decay_l2=0.0001^
 -resnet_n_channels=10 -resnet_block_sizes 3 5 3 -resnet_filter_size 32 64 128 -save_model_epochs 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115