import os
import zipfile
from d2l import torch as d2l

def download_and_save_raw_data(root='../data', filename='raw.txt'):
    os.makedirs(root, exist_ok=True)
    
    zip_path = d2l.download('fra-eng', root)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(root)
    
    fra_txt_path = os.path.join(root, 'fra-eng', 'fra.txt')
    raw_path = os.path.join(root, filename)
    
    with open(fra_txt_path, 'r', encoding='utf-8') as f_in, \
         open(raw_path, 'w', encoding='utf-8') as f_out:
        f_out.write(f_in.read())
    
    print(f'Raw data saved to {raw_path}')
    print('\nFirst 5 lines of the raw data:')
    with open(raw_path, 'r', encoding='utf-8') as f:
        for _ in range(5):
            print(f.readline().strip())

download_and_save_raw_data()
