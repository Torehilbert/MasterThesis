CHANNELS = ['Aperture', 'ApodizedAP', 'BrightField', 'DarkField', 'DFIOpen', 'DFIPhase', 'DPI', 'iSSC', 'Phase', 'UVPhase']


def true_first(i):
    if i==0:
        return True
    else:
        return False


def true_last(i, n_its):
    if i==(n_its-1):
        return True
    else:
        return False


def value_first(i, value, value_alternative=None):
    if i==0:
        return value
    else:
        return value_alternative


def value_last(i, n_its, value, value_alternative=None):
    if i==(n_its-1):
        return value
    else:
        return value_alternative