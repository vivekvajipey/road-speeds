# road-speeds
Predicting road speeds over time using satellite imagery and tabular data

# Installation Instructions

## Step 1: Set Up Python Virtual Environment
1. Ensure you have Python 3.9 installed on your system. If not, install it from the [official Python website](https://www.python.org/).
2. You may need to install the `python3-venv` package to create a virtual environment, which can be done using:

sudo apt install python3-venv

3. Create a virtual environment named `road_env` using:

python3.9 -m venv road_env


## Step 2: Activate the Virtual Environment
1. Activate the virtual environment with:

source road_env/bin/activate # On Unix or MacOS
.\road_env\Scripts\activate # On Windows


## Step 3: Install Required Dependencies
1. Before installing the python packages, some packages require the Python development package and GDAL requires the libgdal library to be pre-installed on your system. Install these using:

sudo apt-get update
sudo apt-get install libgdal-dev
sudo apt install python3.9-dev

2. With the virtual environment activated, install the dependencies (ensure 'requirements.txt' is in the current directory) by running:

pip install -r requirements.txt


## Step 4: Verify Installation
1. You can verify the installation by checking the Python version and installed packages using:

python --version
pip list


# Environment Setup

## Step 1: Enter the Repository
1. Make sure you are in the project repository


## Step 2: Set Up the Python Environment
1. [Follow the installation instructions provided in `installation_requirements.txt` to set up the Conda environment.]
2. Ensure the environment `road_env` is activated before proceeding to the next steps (use: `source road_env/bin/activate`)

## Step 3: Data

In order to generate the 224 x 224 crops of the road segments (which are used as input for the models), the `generate_crops.py` script needs to be run. `generate_crops.py` expects the following directory structure:

PROJECT_ROOT
│\
├── data\
│ ├── GPStracks_forDLcourse.csv # GPS tracks CSV file\
│ ├── img_id_map.json # Generated JSON file (output by script)\
│ ├── crop_info_base.csv # Generated crops info CSV file (output by script)\
│ │\
│ ├── sat_tifs # Directory for TIFF non-sequence images\
│ │ ├── [various TIFF files]\
│ │\
│ ├── sat_tifs_2 # Directory for TIFF sequence images\
│ │ ├── batch2\
│ │ ├── R1_[various TIFF files]\
│ │ ├── R2_[various TIFF files]\
│ │ └── R3_[various TIFF files]\
│ │\
│ ├── rgb_crops_mo_4 # Directory for RGB image crops (output by script)\
│ │ ├── [generated RGB crop files]\
│ │\
│ └── nir_crops_mo_4 # Directory for NIR image crops (output by script)\
│ ├── [generated NIR crop files]\
│\
├── [other project files and directories]\
│\
└── generate_crops.py


The `GPStracks_forDLcourse.csv` is found in the bucket associated with the project data and must be added to the data directory.

Generate the crops by running:

python generate_crops.py


# Experiment Running

## Step 0: Ensure proper environment
1. Before proceeding with running `main.sh`, ensure that the environment is set up as detailed in `installation_requirements.txt` and `environment_setup.txt`.

## Step 1: Running the experiments

1. Running the experiments is done by running the following command in the terminal:

chmod +x main.sh
./main.sh


Here is the directory structure expected by `seq_nn_train.py`:

PROJECT_ROOT\
│\
├── data\
│ ├── crop_info_full_4.csv # The main dataset CSV file\
│ ├── [Other data files]\
│\
├── dataset\
│ ├── init.py\
│ ├── road_mean_std_dataset.py # Contains RoadMeanSTDDataset class\
│ ├── road_mean_std_seq_dataset.py # Contains RoadMeanSTDSeqDataset class\
│ ├── [Other dataset files]\
│\
├── models\
│ ├── init.py\
│ ├── resnet_concat_rnn.py # Contains the ResNetConcatRNN model\
│ ├── [Other model files]\
│\
├── results\
│ ├── models # Directory to save model outputs\
│ │ ├── [Saved model files will be placed here]\
│\
├── transforms.py # Contains data transformations\
│\
├── util.py # Utility functions for the model\
│\
├── crop_info_norm_params.yaml # Normalization parameters\
│\
├── t_cols_norm.yaml # Names of normalized tabular feature columns\
│\
├── training_script.py # The main training script\
│\
└── [Other project files and directories]
