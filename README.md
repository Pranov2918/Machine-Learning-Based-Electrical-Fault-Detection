# Machine Learning-Based Electrical Fault Detection

A machine learning system that uses voltage and current data with time and frequency domain feature engineering to accurately detect electrical faults in power systems, enabling reliable and automated fault diagnosis.

## Motivation

Electrical faults in power systems can cause outages and equipment damage. Early and accurate detection is essential for system reliability and safety. This project automates fault detection using engineered features from electrical signal data.

## Features

- Time and frequency domain feature extraction from voltage and current signals  
- Classification of fault vs no-fault conditions using machine learning  
- Modular Python scripts for easy training, testing, and evaluation  
- Support for custom datasets and real-time fault detection  

## Project Structure

Fault-Detection/
├── data/ # Dataset files e.g., classData.csv
├── src/ # Source code
│ ├── feature_engineering.py
│ ├── train_model.py
│ └── evaluate_model.py
├── models/ # Saved ML models
├── README.md # Project documentation
├── requirements.txt # Python dependencies
└── classData.csv # Sample dataset file


## Installation

Ensure Python 3.6+ is installed. Install required packages:
pip install -r requirements.txt

## Usage Guidelines for Viewers

1. **Prepare Data**  
   Place your dataset in the `data/` folder or use the provided `classData.csv`.

2. **Feature Extraction**  
   Run the feature extraction script:
python src/feature_engineering.py

3. **Train the Model**  
Train the machine learning model with:
python src/train_model.py

4. **Evaluate the Model**  
Test or evaluate model performance:
python src/evaluate_model.py


5. **Customize**  
Modify the Python scripts as needed for your specific dataset or application.

## Results

The model achieves high classification accuracy, providing reliable fault detection for power systems.

## Contributing

Contributions are welcome! Please open issues or pull requests.

## License

MIT License. See the LICENSE file for details.



Guidelines for Viewers
Check your Python version (use Python 3.6 or above).

Install dependencies before running any scripts with pip install -r requirements.txt.

Place data files correctly in the designated data/ folder.

Run scripts in the following order: feature extraction → model training → evaluation.

Review and understand scripts to modify for use with different datasets or scenarios.

Report issues or contribute via GitHub if you find bugs or want to add features.


