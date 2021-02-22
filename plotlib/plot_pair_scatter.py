import matplotlib.pyplot as plt


def plot_pair_scatter(Y1, Y2, show=True, save_path=None):
    # Error checking
    if Y2.shape[1] != Y1.shape[1]:
        raise Exception("Supplied matrices does not have same dimensions")
    
    if Y1.shape[0] != Y2.shape[0]:
        print("WARNING: Supplied matrices does not equal number of datapoints, some will be skipped!")
    n_points = min(Y1.shape[0], Y2.shape[0])
    
    # Subplot logic
    n_plots = Y1.shape[1]
    layout, spots = _get_subplot_layout(Y1.shape[1])
    n_plots = min(Y1.shape[1], spots)
    
    # Create plot
    plt.figure()
    for i in range(n_plots):
        plt.subplot(layout[0], layout[1], i+1)
        plt.scatter(Y1[:n_points,i], Y2[:n_points,i], s=1)
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])
        plt.axis("equal")

    # Save plot
    if save_path is not None:
        plt.savefig(save_path, dpi=500, bbox_inches='tight', pad_inches=0)
    
    # Show plot
    if show:
        plt.show()
    
    # Clean up
    plt.clf()
    plt.close()


def _get_subplot_layout(n):
    if n==0 or n==1:
        return (1,1), 1
    elif n==2:
        return (1,2), 2
    elif n==3:
        return (1,3), 3
    elif n==4:
        return (2,2), 4
    elif n==5 or n==6:
        return (2,3), 6
    elif n>=7 and n<=8:
        return (2,4), 8
    elif n>=9 and n<=12:
        return (3,4), 12
    elif n>=13 and n<=15:
        return (3,5), 15
    elif n>=16 and n<=20:
        return (4,5), 20
    elif n>=21 and n<=25:
        return (5,5), 25
    elif n>=26 and n<=36:
        return (6,6), 36
    elif n>=37 and n<=49:
        return (7,7), 49
    elif n>=50 and n<=64:
        return (8,8), 64
    elif n>=65 and n<=81:
        return (9,9), 81
    elif n>=82 and n<=100:
        return (10,10), 100
    else:
        return (10,10), 100