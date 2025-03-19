Learning objectives
    Determine an appropriate threshold for a binary classification model.
    Calculate and choose appropriate metrics to evaluate a binary classification model.
    Interpret ROC and AUC.

Classification is the task of predicting which of a set of classes (categories) an example belongs to. This module notes how to convert a logistic regression model that predicts a probability into a binary classification model that predicts one of two classes. Also, thi module specifies how to choose and calculate appropiate metrics to evaluate the quality of a classification model;s predictions with a brief introduction to multi-class classification problems, which will later be discussed in more depth.

In order to convert a kigistic regression model's prediction into a category, we need to choose a TRESHOLD PROBABILITY so that 
    Examples with a probability above the threshold value are then assigned to the positive class, the class you are testing for. 
    Examples with a lower probability are assigned to the negative class, the alternative class.
    e.g. : You'd like to deploy this model in an email application to filter spam into a separate mail folder. But to do so, you need to convert the model's raw numerical output (e.g., 0.75) into one of two categories: "spam" or "not spam."

The probability score is not reality, or ground truth, there are 4 possible outcomes for each output from a binary classifier

If you lay out the ground truth as columns and the model's prediction as rows, the following table, called a confusion matrix, is the result:

    | Predicted positive | Actual positive | Actual negative |
    |--------------------|-----------------|-----------------|
    | True positive (TP): A item correctly classified as a item. These are the item messages automatically sent to the item folder. | False positive (FP): A not-item misclassified as an item. These are the legitimate items that wind up in the item folder. |
    | Predicted negative | False negative (FN): A item misclassified as not-item. These are items that aren't caught by the item filter and make their way into the folder. | True negative (TN): A not-item correctly classified as not-item. These are the legitimate items that are sent directly to the folder. |

    Classification:

    True and false positive and negatives are used to calculate several useful metrics for evaluateing models. Which evaluation metrics are most meaningful depends on the specific model and the specific task, the cost of different misclassifications, and wether the dataset is balanced or imbalanced

    Accuracy:
        proportion of all the classifications that were correct, wether positive or negative, mathematically defined as
            acc = correct classifications / total classifications = (tp + tn) / (tp + tn + fp + fn)
        
        a perfect model would have zero false positives and zero false negatives and therefore an accuracy of either 1% or 100%

        however, when the dataset is imbalanced, or where one kind of mistake (Fn or FP) is more costly then the other, which is the case in most real-world applications, it's better to optimizefor one of the metricsinstead.

    Recall, or true positive rate:
        The true positive rate (TPR), or the proportion of all actual positives that were classified correctly as positives, is also known as recall, mathematically defined as :
            TPR = correctly classified actual positives / all actual positived = TP / (TP + FN)

        False negatives are actual posivies that were misclassified as negatives, which is why they appear in the denominator. In the spam classification exampl, recall measures the fraction of spam emails that were correctly classified as spam, this is why another name for recall is probability of detection: it answers the question, What fraction of * is detected?
        
        In an imbalanced dataset where the number of actual positives is very low, recall is a more meaningful metric than accuracy because it measures the ability of the model to correctly identify all positive instances. For applications like disease prediction, correctly identifying the positive cases is crucial. A false negative typically has more serious consequences than a false positive.

    False positive rate:
        the proportion of all actual negatives that were classified incorectly as positives, also known as the probability of false alarm, mathematically defined as
            FPR = incorrectly classified actual negatives / all actual negatives = FP / (FP + TN)
        
        False positives are actual negatives that were misclassified, which is why they appear in the denominator. In the spam classification example, FPR measures the fraction of legitimate emails that were incorrectly classified as spam, or the model's rate of false alarms.

        A perfect model would have zero false positives and therefore a FPR of 0.0, which is to say, a 0% false alarm rate.

        In an imbalanced dataset where the number of actual negatives is very, very low, say 1-2 examples in total, FPR is less meaningful and less useful as a metric.

    Precision:
        the proportion of all the model's positive classifications that are actually positive, mathematically defined as:
            precision = correctly classified actual positives / everything classified as positive = TP / (TP + FP)

        A hypothetical perfect model would have zero false positives and therefore a precision of 1.0.

        In an imbalanced dataset where the number of actual positives is very, very low, say 1-2   examples in total, precision is less meaningful and less useful as a metric.

        Precision improves as false positives decrease, while recall improves when false negatives decrease. But as seen in the previous section, increasing the classification threshold tends to decrease the number of false positives and increase the number of false negatives, while decreasing the threshold has the opposite effects. As a result, precision and recall often show an inverse relationship, where improving one of them worsens the other.

    Choice of metrics and tradeoffs:
        The metric(s) you choose to prioritize when evaluating the model and choosing a threshold depend on the costs, benefits, and risks of the specific problem.

        | Metric                | Guidance                                                                                       |
|-----------------------|-----------------------------------------------------------------------------------------------|
| Accuracy              | Use as a rough indicator of model training progress/convergence for balanced datasets. For model performance, use only in combination with other metrics. Avoid for imbalanced datasets. Consider using another metric. |
| Recall (True positive rate) | Use when false negatives are more expensive than false positives.                              |
| False positive rate   | Use when false positives are more expensive than false negatives.                              |
| Precision             | Use when it's very important for positive predictions to be accurate.                          |


Classification: ROC and AUC
    Evaluate a model's quality across all possible thresholds

    Receiver-operating characteristic curcve ROC:
        The ROC curve is a visual representation of model performance across all thresholds. The long version of the name, receiver operating characteristic, is a holdover from WWII radar detection.

        The ROC curve is drawn by calculating the true positive rate (TPR) and false positive rate (FPR) at every possible threshold (in practice, at selected intervals), then graphing TPR over FPR. A perfect model, which at some threshold has a TPR of 1.0 and a FPR of 0.0, can be represented by either a point at (0, 1) if all other thresholds are ignored, or by the following:

    Area under the curve AUC:
        The area under the ROC curve (AUC) represents the probability that the model, if given a randomly chosen positive and negative example, will rank the positive higher than the negative.

        The perfect model above, containing a square with sides of length 1, has an area under the curve (AUC) of 1.0. This means there is a 100% probability that the model will correctly rank a randomly chosen positive example higher than a randomly chosen negative example. In other words, looking at the spread of data points below, AUC gives the probability that the model will place a randomly chosen square to the right of a randomly chosen circle, independent of where the threshold is set.


Prediction bias:
     calculating prediction bias is a quick check that can flag issues with the model or training data early on.

    Prediction bias is the difference between the mean of a model's predictions and the mean of ground-truth labels in the data. A model trained on a dataset where 5% of the emails are spam should predict, on average, that 5% of the emails it classifies are spam. In other words, the mean of the labels in the ground-truth dataset is 0.05, and the mean of the model's predictions should also be 0.05. If this is the case, the model has zero prediction bias. Of course, the model might still have other problems.

    If the model instead predicts 50% of the time that an email is spam, then something is wrong with the training dataset, the new dataset the model is applied to, or with the model itself. Any significant difference between the two means suggests that the model has some prediction bias.

    Prediction bias can be caused by:
        Biases or noise in the data, including biased sampling for the training set
        Too-strong regularization, meaning that the model was oversimplified and lost some necessary complexity
        Bugs in the model training pipeline
        The set of features provided to the model being insufficient for the task

Multi-class classification:
    Multi-class classification can be treated as an extension of binary classification to more than two classes. If each example can only be assigned to one class, then the classification problem can be handled as a binary classification problem, where one class contains one of the multiple classes, and the other class contains all the other classes put together. The process can then be repeated for each of the original classes.

    For example, in a three-class multi-class classification problem, where you're classifying examples with the labels A, B, and C, you could turn the problem into two separate binary classification problems. First, you might create a binary classifier that categorizes examples using the label A+B and the label C. Then, you could create a second binary classifier that reclassifies the examples that are labeled A+B using the label A and the label B.

    An example of a multi-class problem is a handwriting classifier that takes an image of a handwritten digit and decides which digit, 0-9, is represented.

    If class membership isn't exclusive, which is to say, an example can be assigned to multiple classes, this is known as a multi-label classification problem.