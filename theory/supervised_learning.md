Supervised learning tasks are well-defined and can be applied to a multitude of scenarios, such as identifying spam or predicting precipitation

    Foundational supervised learning concepts

Supervised machine learning is based on the following core concepts:
    *data
        -> the driving force of ml, comes in the form of words and numbers stored in tables, or as the value of pixels and waveforms captured in images and audio files,
        -> stored in datasets
            -datasets are made up of individual examples that contain features and a label (features are the values used to predict the label)

    *model
        -> a model is the complex collection of numbers that define the mathematical relationship from specific input feature patterns to specific output label values, the model discovers those patterns through training.

    *training
        -> Before a supervised model can make predictions it must be trained.
        -> to train a model we give the model a dataset with labeled examples
        -> the model learns the mathematical relationship between the features and the label so that it can make the best preductions

    *evaluating
        -> We evaluate a trained model to determine how well it learned, when we evaluate a model we use a labeled dataset but we only give the model the dataset;s features

    *inference
        -> Once we are satisfied with the results from evaluateing the model, we can use the model to make predictions, called INFERENCE, on unlabeled examples.
