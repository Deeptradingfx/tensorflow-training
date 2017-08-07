import numpy as np

SEQ_LEN = 10

def create_time_series():
    freq = (np.random.random() * 0.5) + 0.1 
    ampl = np.random.random() + 0.5
    x = np.sin(np.arrange(0, SEQ_LEN) * freq) * ampl
    return x
