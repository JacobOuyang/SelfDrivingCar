import os
import random
from shutil import copyfile
def read_file_names(src_folder):
    file_names = []
    for root,dirs,files in os.walk(src_folder, followlinks=True):
        for file in files:
            file_names.append([os.path.join(root, file), file])

    return file_names

def upsample_files(file_names, dst_folder, index):

    for file_name in file_names:
        for i in range(index):
            copyfile(file_name[0], os.path.join(dst_folder, str.format("%d_%s"%(i,file_name[1]))))
def main():


    dst_folder = "/home/jacob/Desktop/trainingfolder/DirL"
    if not os.path.exists(dst_folder):
        os.makedirs(
            dst_folder
        )
    file_names = read_file_names("/home/jacob/Desktop/trainingfolder/DirL")
    upsample_files(file_names, dst_folder, 2)
"""   dst_folder = "/home/jacob/Desktop/trainingfolder/DirR"
    if not os.path.exists(dst_folder):
        os.makedirs(
            dst_folder
        )
    file_names = read_file_names("/home/jacob/Desktop/trainingfolder/DirR")
    upsample_files(file_names, dst_folder, 2)
    """
if __name__ == "__main__":
    main()