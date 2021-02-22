import matplotlib.pyplot as plt
import seaborn as sns
import os




def plot_line(x, y, xlabel=None, ylabel=None, label=None, fig=None, xlog=False, ylog=False, call_legend=False, tight_layout=False, show=False, save_configs={}, figsize=(6,4), **kwargs):
    sns.set(style='whitegrid')

    # construct figure
    if fig is None:
        fig = plt.figure(figsize=figsize)
    
    # plotting
    plt.plot(x, y, **kwargs, label=label)

    # axis labels
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if xlog:
        plt.xscale('log')
    if ylog:
        plt.yscale('log')
    if call_legend:
        plt.legend()
    if tight_layout:
        plt.tight_layout()

    # save
    if save_configs is not None and len(save_configs) > 0:
        root = save_configs['root'] if 'root' in save_configs else None
        filename = save_configs['filename'] if 'filename' in save_configs else 'noname'
        dpi = save_configs['dpi'] if 'dpi' in save_configs else 250
        formats = save_configs['format'] if 'format' in save_configs else 'png'
        if type(formats) is not list and type(formats) is not tuple:
            formats = [formats]

        for fmt in formats:
            filename_ext = filename + "." + fmt
            path_file = os.path.join(root, filename_ext) if root is not None else filename_ext
            plt.savefig(path_file, dpi=dpi)
    
    # show
    if show:
        plt.show()
    
    return fig



if __name__ == "__main__":
    x= [1,2,3,4]
    y = [1,4,9,16]

    fig = plot_line(
        x=x,
        y=y, 
        xlabel='my x label', 
        ylabel='my y label', 
        label='my curve', 
        call_legend=False, 
        )

    plot_line(
        x=x,
        y=[2,5,12,20], 
        xlabel='my x label', 
        ylabel='my y label', 
        label='my curve2', 
        call_legend=True, 
        show=True, 
        fig=fig,
        tight_layout=True,
        save_configs={
            'format':['png', 'pdf'],
            'filename':'myplot',
            'dpi':250}
            )

    # plt.plot(x,y)
    # plt.xlabel('what')
    # plt.ylabel('what')
    # plt.legend()
    # plt.show()