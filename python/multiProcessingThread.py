from PIL import Image
import concurrent.futures
import time
import os

def process_img(filename):
    """
    This method take the filename of the img file, performs a gray scale operation on the file, creates
    a thumbnail of the img and then saves the file in processed folder.
    :param filename: name of img file.
    :return: None
    """
    pathInput = ".\\imgs\\"
    pathOutput = ".\\processed\\"
    img = Image.open(pathInput + filename)
    gray_img = img.convert('L')
    gray_img.thumbnail((200, 400))
    gray_img.save(pathOutput + filename)

# Mehtods that perform extensive I/O operations.
# @calculate_exe_time
def io_operations_without_concurrency(file_names):
    """
    This method calls the process_img method for each file which performs simple image processing 
    on the img file and then moves the files to processed folder. This is done in a regular fashion 
    without any concurrency.
    :param file_names: list containing names of the img files to process
    :return: None
    """

    for filename in file_names:
        # startTime = time.time()
        process_img(filename)
        # print("{}: {}".format(filename, time.time() - startTime))


# @calculate_exe_time
def io_operations_via_threading(file_names):
    """
    This method calls the process_img method for each file which performs simple image processing 
    on the img file and then moves the files to processed folder. This is done in a concurrent fashion
    using the multithreading feature in concurrent.futures module.
    :param file_names: list containing names of the img files to process
    :return: None
    """

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_img, file_names)


# @calculate_exe_time
def io_operations_via_multiprocessing(file_names):
    """
    This method calls the process_img method for each file which performs simple image processing 
    on the img file and then moves the files to processed folder. This is done in a concurrent fashion using the multiprocessing feature in 
    concurrent.futures module.
    :param file_names: list containing names of the img files to process
    :return: None
    """

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process_img, file_names)

file_names = os.listdir(".\\imgs\\")
# print(file_names)
# for filename in file_names:
#     print(filename)

# file_names = {"Mango_01_A.JPG", "Mango_01_B.JPG", "Mango_02_A.JPG", "Mango_02_B.JPG",\
#             "Mango_03_A.JPG", "Mango_03_B.JPG", "Mango_04_A.JPG", "Mango_04_B.JPG",\
#             "Mango_05_A.JPG", "Mango_05_B.JPG"}
# print(file_names)
# for filename in file_names:
#     print(filename)

# startTimeSum = time.time()
# io_operations_without_concurrency(file_names)
# print("[INFOR]: Time io_operations_without_concurrency: {}".format(time.time() - startTimeSum))
# print("----------------------------------DONE---------------------------------")

# startTimeSum = time.time()
# io_operations_via_threading(file_names)
# print("[INFOR]: Time io_operations_via_threading: {}".format(time.time() - startTimeSum))
# print("----------------------------------DONE---------------------------------")

# startTimeSum = time.time()
# io_operations_via_multiprocessing(file_names)
# print("[INFOR]: Time io_operations_via_multiprocessing: {}".format(time.time() - startTimeSum))
# print("----------------------------------DONE---------------------------------")