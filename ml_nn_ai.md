The main diffrence inbetween artificial inteligence, neural networks and machine learning and how to differentiate in-between them.

    {AI field   [ML Field   (NN field   )   ]   }

    *ai -> the effort to automate intellectual tasks normally efectuated by humans
    
    *ml -> take the data, figure out what is good/bad, figure feed data

    classical programming:
        data  -> |
                 | -> answers
        rules -> |

    ml programming:
        data    -> |
                   | -> rules
        answers -> |

        goal is to get the best accuracy as possible

    *nn (deep learning) -> a form of machine that uses a layered representation of data


Tensors ->
    "A tensor is a generalization of vectors and matrices to potentially higher dimensions. Internally, TensorFlow represents tensors as n-dimensional arrays of base datatypes." (https://www.tensorflow.org/guide/tensor)

    It should't surprise you that tensors are a fundemental apsect of TensorFlow. They are the main objects that are passed around and manipluated throughout the program. Each tensor represents a partialy defined computation that will eventually produce a value. TensorFlow programs work by building a graph of Tensor objects that details how tensors are related. Running different parts of the graph allow results to be generated.

    Each tensor has a data type and a shape.

    Data Types Include: float32, int32, string and others.

    Shape: Represents the dimension of data.

    Just like vectors and matrices tensors can have operations applied to them like addition, subtraction, dot product, cross product etc.


