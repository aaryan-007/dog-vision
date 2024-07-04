# 🐶 Using Transfer Learning and TensorFlow 2.0 to Classify Different Dog Breeds

Dogs are incredible. But have you ever been sitting at a cafe, seen a dog and not known what breed it is? I have. And then someone says, "it's an English Terrier" and you think, how did they know that?

In this project, we're going to be using machine learning to help us identify different breeds of dogs.

To do this, we'll be using data from the [Kaggle dog breed identification competition](https://www.kaggle.com/c/dog-breed-identification/overview). It consists of a collection of 10,000+ labeled images of 120 different dog breeds.

This kind of problem is called multi-class image classification. It's multi-class because we're trying to classify multiple different breeds of dog. If we were only trying to classify dogs versus cats, it would be called binary classification (one thing versus another).

Multi-class image classification is an important problem because it's the same kind of technology Tesla uses in their self-driving cars or Airbnb uses in automatically adding information to their listings.

Since the most important step in a deep learning problem is getting the data ready (turning it into numbers), that's what we're going to start with.

## Workflow

We're going to go through the following TensorFlow/Deep Learning workflow:
1. **Get data ready**: Download from Kaggle, store, and import.
2. **Prepare the data**: Preprocessing, the 3 sets, X & y.
3. **Choose and fit/train a model**: Using [TensorFlow Hub](https://www.tensorflow.org/hub), `tf.keras.applications`, [TensorBoard](https://www.tensorflow.org/tensorboard), [EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping).
4. **Evaluate the model**: Making predictions and comparing them with the ground truth labels.
5. **Improve the model through experimentation**: Start with 1000 images, ensure it works, then increase the number of images.
6. **Save, share, and reload your model**: Once you're happy with the results.

## Project Structure

- `data/`: Contains the dataset used for training and evaluation.
- `notebooks/`: Jupyter notebooks used for experimentation and model development.
- `src/`: Source code for the project, including data preprocessing, model training, and evaluation scripts.
- `models/`: Saved models and checkpoints.
- `logs/`: TensorBoard logs for visualizing training progress.

## Getting Started

### Prerequisites

- Python 3.6 or later
- TensorFlow 2.0 or later
- [Kaggle API](https://github.com/Kaggle/kaggle-api) for downloading the dataset

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/dog-vision-project.git
    cd dog-vision-project
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset from Kaggle:
    ```bash
    kaggle competitions download -c dog-breed-identification
    unzip dog-breed-identification.zip -d data/
    ```

### Usage

Run the Jupyter notebooks in the `notebooks/` directory to see the step-by-step process of data preparation, model training, and evaluation.

## Results

- Achieved an accuracy of 95.31% on the test set using transfer learning with a pre-trained model.


## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.


