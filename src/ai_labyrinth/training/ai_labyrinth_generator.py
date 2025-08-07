import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping
from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
from ai_labyrinth.ai.model import unet
from create_dataset import draw_maze
#Let's create a function for one step of the encoder block, so as to increase the reusability when making custom unets

def training(model: Model, X_train: np.ndarray, y_train: np.ndarray, val_data: np.ndarray) -> None:
  
  model.compile(loss = 'categorical_crossentropy',
            optimizer = 'adam',
            metrics = ['accuracy'])

  #Defining early stopping to regularize the model and prevent overfitting
  early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3, restore_best_weights = True)

  #Training the model with 50 epochs (it will stop training in between because of early stopping)
  unet_history = model.fit(x=X_train, y=y_train, validation_data = val_data, 
                        epochs = 50, callbacks = [early_stopping])
  
  model.save_weights('unet_labyrinth.weights.h5')

def create_train_val_test_sets(data_path: str, test_size: float=0.2, val_size: float=0.2, random_state: int=42) -> tuple[int, int, int, int, int, int]:
    data = np.load(data_path)
    X = data['inputs']
    y = data['outputs']
    # Étape 1: Séparation en entraînement+validation et test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Étape 2: Séparation de l'ensemble entraînement+validation en entraînement et validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def post_process_prediction(prediction: np.ndarray) -> np.ndarray:
    clean_maze = np.argmax(prediction, axis=-1)

    return clean_maze

def testing(model: Model, X_test: np.ndarray) -> None:
        model.load_weights('unet_labyrinth.weights.h5')
        prediction = model.predict(X_test)
        clean_maze = post_process_prediction(prediction)
        for maze in clean_maze:
           draw_maze(maze)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script pour entraîner ou tester un modèle U-Net pour la génération de labyrinthes.")
    parser.add_argument('action', choices=['train', 'test'], help="L'action à effectuer : 'train' pour l'entraînement, 'test' pour le test.")
    
    args = parser.parse_args()
    
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_sets('dataset_labyrinthes.npz')

    model = unet()
    
    if args.action == 'train':
        training(model, X_train, y_train, (X_val, y_val))
    elif args.action == 'test':
        testing(model, X_test)
