import os
import random
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
def balance_data(data_filepath):
    '''Balances image dataset (e.g., NORMAL vs PNEUMONIA) by undersampling the majority class.
    
    Parameters:
        data_filepath (str): Path to the dataset directory containing 'NORMAL' and 'PNEUMONIA' subfolders.

    Returns:
        df: A balanced DataFrame with two columns:
            - 'filename': full file paths to images
            - 'class': corresponding class labels ('NORMAL' or 'PNEUMONIA')
    
    '''
    # Load all image paths and labels
    image_paths = []
    labels = []
    for class_name in ["NORMAL", "PNEUMONIA"]:
        class_path = os.path.join(data_filepath, class_name)
        for image_name in os.listdir(class_path):
            image_paths.append(os.path.join(class_path, image_name))
            labels.append(class_name)

    # Separate paths by class
    normal_paths = [path for path, label in zip(image_paths, labels) if label == "NORMAL"]
    pneumonia_paths = [path for path, label in zip(image_paths, labels) if label == "PNEUMONIA"]

    # Undersample the majority class (PNEUMONIA)
    num_normal = len(normal_paths)
    pneumonia_paths = random.sample(pneumonia_paths, num_normal)  # Undersample to match NORMAL count

    # Combine paths and labels for the balanced dataset
    balanced_paths = normal_paths + pneumonia_paths
    balanced_labels = ['NORMAL'] * num_normal + ['PNEUMONIA'] * num_normal

    return pd.DataFrame({'filename': balanced_paths, 'class': balanced_labels})


def apply_gaussian_blur(img):
    '''Apply the Ben Graham Gaussian blur enhancement method to a grayscale image.
    This technique enhances image contrast and sharpness by subtracting a heavily
    blurred version of the image from the original, creating a high-pass filter effect.

    Parameters:
        img (np.ndarray): Grayscale image array (H, W) or (H, W, 1).

    Returns:
        np.ndarray: Preprocessed image with shape (H, W, 1) and dtype float32.
    '''
    img = img.astype(np.uint8)  # Convert to uint8 for OpenCV operations
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 512 / 10), -4, 128)
    img = img.astype(np.float32)  # Convert back to float for rescaling
    img = np.expand_dims(img, axis=-1)  # Add channel dimension: (H, W) -> (H, W, 1)
    return img

def load_dataset_from_dir(data_filepath, img_size=(128, 128), batch_size=16, gaussian_blur=False):
    '''Load image data from directory, applies optional preprocessing (Gaussian blur), and returns a 
    Keras DirectoryIterator for model training or evaluation.

    Parameters:
        data_filepath (str): Path to the directory containing subdirectories of image classes.
        img_size (tuple): Target size for resizing images (height, width).
        batch_size (int): Number of images per batch.
        gaussian_blur (bool): If True, apply method to enhance images.

    Returns:
        dataset (DirectoryIterator): Iterator that yields batches of preprocessed images and labels.
    '''
    # Preprocessing function to apply a gaussian_blur
    preprocessing_func = apply_gaussian_blur if gaussian_blur else None
    
    # Create an ImageDataGenerator instance with desired preprocessing steps
    datagen = ImageDataGenerator(
      preprocessing_function=preprocessing_func,
      rescale=1./255,
      shear_range=0.3,
      horizontal_flip=True,
      brightness_range=[0.7, 1.3]
  )

    # Create a flow_from_directory for the dataset
    dataset = datagen.flow_from_directory(
        data_filepath,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        color_mode='grayscale',
      )

    return dataset


def load_dataset_from_df(df, filename_column, label_column, img_size=(128, 128), batch_size=16, gaussian_blur=False):
    '''Load image data from a dataframe, applies optional preprocessing (Gaussian blur), and returns a 
    Keras DirectoryIterator for model training or evaluation.

    Parameters:
        df: dataframe containing the filepaths and class labels
        img_size (tuple): Target size for resizing images (height, width).
        batch_size (int): Number of images per batch.
        gaussian_blur (bool): If True, apply method to enhance images.

    Returns:
        dataset (DirectoryIterator): Iterator that yields batches of preprocessed images and labels.
    '''
    # Preprocessing function to apply a gaussian_blur
    preprocessing_func = apply_gaussian_blur if gaussian_blur else None
    
    # Create an ImageDataGenerator instance with desired preprocessing steps
    datagen = ImageDataGenerator(
      preprocessing_function=preprocessing_func,
      rescale=1./255,
      shear_range=0.3,
      horizontal_flip=True,
      brightness_range=[0.7, 1.3]
  )

    # Create flow_from_dataframe using balanced data
    dataset = datagen.flow_from_dataframe(
        dataframe=df,
        directory=None,
        x_col=filename_column,
        y_col=label_column,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        color_mode='grayscale'
    )

    return dataset

def load_sample_dataframe_from_dir(df, data_filepath, class_label, sample_size, seed):
    ''' Appends a random sample of image paths and their class label to the input dataframe

    Parameters:
        df (pd.DataFrame): Existing dataframe to append to.
        data_filepath (str): Directory containing image files.
        class_label (str): Class label to associate with each image.
        sample_size (int): Number of images to randomly sample.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Updated dataframe with new image entries.
    '''
    new_rows = []

    for image in os.listdir(data_filepath):
        image_path = os.path.join(data_filepath, image)
        if os.path.isfile(image_path):
            new_rows.append({'filename': image_path, 'class': class_label})

    # Convert new rows to DataFrame and concatenate
    new_df = pd.DataFrame(new_rows)
    new_df_sampled = new_df.sample(n=sample_size, random_state=seed)
    df = pd.concat([df, new_df_sampled], ignore_index=True)

    return df
    