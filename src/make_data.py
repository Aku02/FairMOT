import os
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/kaggle/input/nfl-data-yolo/dataset')

    args = parser.parse_args()

    SRC_DIR = args.data_dir
    TRAIN_IMG_DIR = f'{SRC_DIR}/train/images'
    TEST_IMG_DIR = f'{SRC_DIR}/valid/images'

    # Step 1. Generate files containing image paths and save to src/data
    train_file = 'src/data/nfl.train'
    test_file = 'src/data/nfl.val'
    train_imgs = os.listdir(TRAIN_IMG_DIR)  # not sort yet
    test_imgs = os.listdir(TEST_IMG_DIR)  # not sort yet

    with open(train_file, 'w') as f:
        f.write("\n".join(train_imgs))
    with open(test_file, 'w') as f:
        f.write("\n".join(test_imgs))
        
    # Step 2. Create a json file for the dataset in src/lib/cfg/
    cfg = dict(
        root=TRAIN_IMG_DIR,
        train=dict(nfl=train_file),
        test=dict(nfl=test_file),
        test_emb=dict(nfl=test_file),
    )
    with open('src/lib/cfg/nfl_data.json', 'w') as f:
        json.dump(cfg, f, indent=4)