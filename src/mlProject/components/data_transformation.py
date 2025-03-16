import os
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from mlProject.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up


    def train_test_spliting(self):
        # Load data
        data = pd.read_csv(self.config.data_path)

        # Remove the 'Id' column
        if 'Id' in data.columns:
            data.drop(columns=['Id'], inplace=True)
            logger.info("Removed 'Id' column from dataset.")

        # Split the data into training and test sets (0.75, 0.25 split)
        train, test = train_test_split(data, test_size=0.25, random_state=42)

        # Save train and test data
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        # Logging information
        logger.info("Split data into training and test sets")
        logger.info(f"Train Shape: {train.shape}")
        logger.info(f"Test Shape: {test.shape}")

        print("Train Shape:", train.shape)
        print("Test Shape:", test.shape)