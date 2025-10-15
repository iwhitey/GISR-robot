import sys, os
import random
import shutil

data_path = ""
destination_folder = ""
num_samples = 500

files = os.listdir(data_path)
files.remove('_camera_settings.json')
json_files = [file for file in files if ".json" in file]
num_files = len(json_files)
copied_files = []

for idx, file in enumerate(json_files):
    # with open(os.path.join('samples/single_sample_panda/002424.json')) as f:
    data_id = file.split('.')[0]
    file_idx = random.randint(0, num_files)

cntr = 0
while True:
    if cntr == num_samples: break

    file_idx = random.randint(0, num_files)
    if file_idx in copied_files: continue
    copied_files.append(file_idx)

    json_file = json_files[file_idx]
    img_file = json_file.split('.')[0] + ".rgb.jpg"

    json_path = os.path.join(data_path, json_file)
    img_path = os.path.join(data_path, img_file)
    print(f"Copying {cntr}/{num_samples} files")
    shutil.copy(json_path, destination_folder)
    shutil.copy(img_path, destination_folder)
    cntr += 1
    
print(f"Done, copied {cntr} files to {destination_folder}")
