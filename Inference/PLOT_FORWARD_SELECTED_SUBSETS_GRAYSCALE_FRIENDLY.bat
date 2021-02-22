python plot_accuracy_by_channel_subset.py^
 -subset_series^
 "2-24-246-2468-02478-024678-0245678-01245678-023456789-0123456789"^
 "RANDOM_1-RANDOM_2-RANDOM_3"^
 "6-26-246-2468-24568-124568-1245678-01245678-012456789"^
 "6-26-256-2356"^
 "9-09-089-0689-01689-014689-0145689-01245689-012456789"^
 "4-48-048-0478-02478-024678-0245678-02345678-023456789"^
 -colors "black" "lightgray" "tan" "darkgreen" "steelblue" "salmon"^
 -legends "Ground truth" "Random" "Entropy" "Correlation" "Channel occlusion" "Teacher-student"^
 -xlabel="Num. selected channels"^
 -xtickadd=1^
 -figsize 7.4 4.0^
 -ylim 0.9275 0.975^
 -fmt "- -x :D -.s --^ -o"^
 -err_linewidths 1 1 1.5 1.5 1.5 1.5^
 -linewidths 1 1 2.5 2.5 2.5 2.5^
 -markersizes 10.0 10.0 7.5 7.5 9.0 7.5^
 -save=1
pause