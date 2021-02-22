python plot_accuracy_by_channel_subset.py^
 -method=minmax^
 -subset_series^
 "2-24-247-2479"^
 "RANDOM_1-RANDOM_2-RANDOM_3-RANDOM_4"^
 "6-26-246-2468-24568-124568-1245678-01245678-012456789"^
 "6-26-256-2356-2568"^
 "9-09-089-0689-01689-014689-0145689-01245689-012456789"^
 "4-48-048-0478-02478-024678-0245678-02345678-023456789"^
 "0-04-025-0235-14678-146789-1346789-12346789-123456789"^
 -colors "black" "lightgray" "tan" "darkgreen" "steelblue" "salmon" "orchid"^
 -fmt "-- -- - - - - -"^
 -legends "Best selection" "Random" "Entropy" "Correlation" "Channel occlusion" "Teacher-student" "Deep subspace clustering"^
 -also_look_in_random_folders 1 0 0 0 0 0 0^
 -xlabel="Num. selected channels"^
 -xtickadd=1^
 -figsize 6 4^
 -ylim 0.9275 0.975^
 -err_linewidths 0.75 0.75 0.75 0.75 0.75 0.75 0.75^
 -linewidths 1 1 2.5 2.5 2.5 2.5 2.5^
 -markersizes 10.0 10.0 7.5 7.5 9.0 7.5 7.5^
 -save=1
pause