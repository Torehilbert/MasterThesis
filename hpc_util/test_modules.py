

if __name__ == "__main__":
    modules = {}

    # Numpy
    try:
        import numpy
        modules['numpy'] = 1
    except:
        modules['numpy'] = 0

    # Pandas
    try:
        import pandas
        modules['pandas'] = 1
    except:
        modules['pandas'] = 0

     # Matplotlib
    try:
        import matplotlib
        modules['matplotlib'] = 1
    except:
        modules['matplotlib'] = 0

    # Tensorflow
    try:
        import tensorflow
        modules['tensorflow'] = 1
    except:
        modules['tensorflow'] = 0

    # Print results:
    for key,val in modules.items():
        print(key, val)
    
    