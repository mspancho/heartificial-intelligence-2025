# Classifying Chagas Disease: Electrocardiogram Bogaloo
Chagas disease is a parasitic illness caused by the protozoan Trypanosoma cruzi, primarily induced by triatomine bugs. It is endemic to Central and South America but can also be found in other parts of the world through migration. 
The disease begins acutely with fever, fatigue, and swelling near the infection site.
However, about 20–30% of infected individuals may develop chronic Chagas cardiomyopathies and require intensive care.
Chagas disease has ben hypothesized to be detectable on a 12-lead ECG, and a fast deep learning model has the potential to promote widespread preliminary testing, efficient treatment, and vector control.

In response to the George B. Moody Physionet Challenge for 2025, this is our attempt to create a diagnostic model that classifies Chagas disease from 12 lead ECG data using a convolutional neural network.

We did this project as a final for our course CSCI1470 at Brown university. Please see the poster, write up, and some technical details below. Special thanks to Professor Eric Ewing for a great semester!


## Poster
![heartificial_intelligence_dlday](https://github.com/user-attachments/assets/1295c658-48af-4615-9c6f-1b05d8cc3eca)


## Write Up
For those interested, please take a look at an indepth write up of our project [linked here](https://docs.google.com/document/d/1HzPNSQLeeYuLWzl6L6fi1B5jrJpqsSR4f7r5lWXN2Nk/edit?usp=sharing)



## CODE-15% dataset

In accordance with the rules of physionet, we used the CODE-15% dataset, here are some instructions for downloading and preprocessing the data

These instructions use `code15_input` as the path for the input data files and `code15_output` for the output data files, but you can replace them with the absolute or relative paths for the files on your machine.

1. Download and unzip one or more of the `exam_part` files and the `exams.csv` file in the [CODE-15% dataset](https://zenodo.org/records/4916206).

2. Download and unzip the Chagas labels, i.e., the [`code15_chagas_labels.csv`](https://physionetchallenges.org/2025/data/code15_chagas_labels.zip) file.

3. Convert the CODE-15% dataset to WFDB format, with the available demographics information and Chagas labels in the WFDB header file, by running
        python prepare_code15_data.py \
            -i code15_input/exams_part0.hdf5 code15_input/exams_part1.hdf5 \
            -d code15_input/exams.csv \
            -l code15_input/code15_chagas_labels.csv \
            -o code15_output/exams_part0 code15_output/exams_part1

Each `exam_part` file in the [CODE-15% dataset](https://zenodo.org/records/4916206) contains approximately 20,000 ECG recordings. You can include more or fewer of these files to increase or decrease the number of ECG recordings, respectively. You may want to start with fewer ECG recordings to debug your code.

## File Structure
After downloading and preprocessing the data we set up a data file structure with a test_data folder and train_data folder in our working directory.
We partitioned the data manually into a 75:25 train test split in these folders. As a result of the size of our dataset, we chose to do this for considerations of local memory.
With more compute, it would be more ideal to randomize the train/test split every time to obtain more representative results.


## Contact Us!
Intrigued? Please email stephen_c_yang@brown.edu and brandon_lien@brown.edu with any questions or comments :)
