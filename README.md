# Areca Disease Prediction (ResNet)

This project implements a deep learning model using ResNet architecture to predict diseases in areca (betel nut) plants. The model classifies various diseases and healthy conditions based on images of nuts, trunks, leaves, and other plant parts.

## Features

- Disease classification for areca plants
- Uses ResNet architecture for high accuracy
- Supports multiple disease types including Mahali Koleroga, stem cracking, stem bleeding, yellow leaf disease, and bud borer
- Includes healthy classifications for nuts, trunks, leaves, and foot

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/k-raiiiz/areca-disease-prediction.git
   cd areca-disease-prediction
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. Install the required packages:
   ```
   pip install -r Requirements.txt
   ```

## Usage

1. Ensure your dataset is in the `Arecanut_dataset/` folder (note: this folder is gitignored by default due to size)

2. Run the application:
   ```
   python App.py
   ```

3. For model training or evaluation, refer to `Model.ipynb` notebook

## Dataset

The dataset should be organized as follows:
```
Arecanut_dataset/
├── train/
│   ├── Healthy_Nut/
│   ├── Healthy_Trunk/
│   ├── Mahali_Koleroga/
│   └── ...
├── test/
│   └── ...
└── final_testing/
    └── ...
```

**Note:** The dataset folder is excluded from version control due to its large size. You'll need to add your dataset manually.

## Model Files

Pre-trained models are included:
- `areca_FINAL_CHAMPION.h5` - Best performing model
- `areca_95_PERCENT_CHAMPION.h5` - 95% accuracy model
- Other model variants for different stages

## Requirements

See `Requirements.txt` for all dependencies.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

