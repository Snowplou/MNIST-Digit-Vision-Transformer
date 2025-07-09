## MNIST DIGIT VISION TRANSFORMER
This project utilizes a vision transformer to recognize digits from the MNIST Digit dataset. The model size is easily adjustable in [config.py](config.py).

# [Interactive.py](interactive.py)
This file allows you to draw digits on the screen and view the outputted predictions from the trained model.

# [Predict.py](predict.py)
Gets the model's predictions for the test dataset from the [Kaggle competition](https://www.kaggle.com/competitions/digit-recognizer) to test the model's accuracy.

# [Run.py](run.py)
Selects an image from the train dataset and outputs the predicted and true label to better understand what mistakes the AI is making on the data it was trained on.

# [Train.py](train.py)
Trains the vision transformer and saves the model. The model already provided in this repository was trained on the dataset from this [Kaggle competition](https://www.kaggle.com/competitions/digit-recognizer).
