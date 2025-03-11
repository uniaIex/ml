Learning objectives:
    -understand diffrent types of machine learning
    -understand key concepts of supervised machine learning
    -learn how solving problems with ml is diffrent from traditional approaches

ml is the process of training a piece of software, called a model, to make useful predictions, or generate content from data

types of ml systems (based on how it learns to do predictions)
    -supervised learning
    -unbsupervised learning
    -reinforcement learning
    -generative ai


    Supervised learning
        -models make predictions after seeing lots of data with the correct answers and then discovers the connections between the elements in the data that produce the correct answers
        -most common use cases for supervised learning are regression and classification

        *regression -> a regression model predicts a numerical value, e.x. a weather model that predicts the amoun of rain
        *classification -> classification models predict the likelyhood that something belongs to a particular category and outputs a value that state so
            -divided in 2 groups:
                *binary classification - models output a value from a class that contains only 2 values (e.x. a model that outputs either rain or no rain)
                *multiclass classification - models output a value from a class that contains more then 2 values, (e.x. a model that can output either rain, hail, snow or sleet)

    Unsupervised learning
        -models make predictions by being given data that does not contain any correct answers.
        -the models goal is to identify meaningful patterns among the data\
        -the model has no hints on howe to categorize each piece of data, but instead it must infer its own rules
        -a commonly used unsupervised learning model employs a technique called clustering, the model finds data points that demarcate natural groupings

    Reinforcement learning
        -models make predictions by getting rewards or penalties based on actions performed with an environment.
        -generates a policy that defines the best stratefy for getting the most rewards
        -used to train robots to perform tasks, like walking around a room, and software programs to play games

    Generative ai
        -class of models that creates content from user input
        -