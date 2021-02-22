from matplotlib.colors import LinearSegmentedColormap

COLORS = ['royalblue', 'limegreen', 'indianred', 'orange', 'red', 'yellow', 'blue', 'blue', 'blue', 'blue', 'blue']
COLORS_NO_CLASS = [(0,0,0),(0.2,0.2,0.2),(0.4,0.4,0.4)]
COLOR_POSITIVE = 'indianred'
COLOR_NEUTRAL = 'whitesmoke'
COLOR_NEGATIVE = 'royalblue'
SCATTER_MARKER_SIZE = 5
COLORS_METHOD = ["darkgray", "steelblue", "salmon"]

CMAP_NEGPOS = LinearSegmentedColormap.from_list("POSNEG", colors=[COLOR_NEGATIVE, COLOR_NEUTRAL, COLOR_POSITIVE])