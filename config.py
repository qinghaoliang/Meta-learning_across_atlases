import os

class config:
    BASE_DIR = './' 
    IN_DIR = os.path.join(BASE_DIR, 'input_dnn')
    OUT_DIR = os.path.join(BASE_DIR, 'output_dnn')

    BATCH_SIZE = 128
    EPOCHS = 1000
    RUNS = 1
    RAMDOM_SEED = 1
    KS = [10, 20, 50, 100, 200]
