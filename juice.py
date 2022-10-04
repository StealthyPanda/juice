from typing import List, Callable, Tuple
import numpy as np
from math import exp, tanh, e


alpha = 0.5

actnone =  np.vectorize(lambda x: x)
actsigmoid = np.vectorize(lambda x: (1/(1+pow(e, -x))))
actrelu = np.vectorize(lambda x: (alpha*x) if x >= 0 else 0)
actswish = np.vectorize(lambda x: (x/(1+pow(e, -x))))
acttanh = np.vectorize(lambda x: tanh(x))
actelu = np.vectorize(lambda x: x if x >= 0 else (alpha * (pow(e, x) - 1)))
actthrow = np.vectorize(lambda x : pow(x, 2) * pow(e, x))


dernone =  np.vectorize(lambda x: 1)
dersigmoid = np.vectorize(lambda x: (pow(e, x)/pow((1+pow(e, x)), 2)))
derrelu = np.vectorize(lambda x: (alpha) if x >= 0 else 0)
derswish = np.vectorize(lambda x: pow(e, x) * ((x + 1 + pow(e, x))/pow((1+pow(e, x)), 2)))
dertanh = np.vectorize(lambda x: (1 - pow(tanh(x), 2)))
derelu = np.vectorize(lambda x: 1 if x >= 0 else (alpha * (pow(e, x))))
derthrow = np.vectorize(lambda x : pow(e, x) * (pow(x, 2) + (2 * x)))

derivatives = {
    actnone : dernone,
    actswish : derswish,
    actrelu : derrelu,
    actsigmoid : dersigmoid,
    actelu : derelu,
    actthrow : derthrow
}


def Generative1D(
    
    x : List[float],
    w  : List[List[float]],
    activations : List[Callable[[float], float]] = [actnone, actnone],
    form : Callable = None
    
    ) -> List[float]:
    
    if form is None:
        return [
            activations[0]((w[0][0] * x) + w[0][1]),
            activations[1]((w[1][0] * x) + w[1][1])
        ]
    else: return form(x, w)


class Layer:

    def __init__(self, cells : int, inputsize : int = None, activation : Callable[[float], float] = actnone, random : bool = False, name : str = None) -> None:
        self.default = np.random.random if random else np.ones
        self.name = 'Dense Layer' if name is None else name
        self.inputsize = inputsize
        self.cells = cells
        if inputsize is not None: self.array = self.default((cells, inputsize + 1))
        self.activation = activation
    
    def __mul__(self, other : np.ndarray) -> np.ndarray:
        assert self.inputsize is not None, "Layer hasn't been assigned input size yet!"
        if len(other.shape) > 1:
            raw = np.matmul(self.array, np.concatenate((other, [[1 for i in range(other.shape[1])]]), axis = 0))
        else:
            raw = np.matmul(self.array, np.concatenate((other, [1])))
        return raw
    
    def spawn(self) -> None:
        self.array = self.default((self.cells, self.inputsize + 1))

class Flatten:

    def __init__(self, inputsize : Tuple[int, int]) -> None:
        self.name = 'Flatten Layer'
        self.inputsize = inputsize
        self.activation = actnone

        self.cells = 1

        for each in range(len(inputsize)):
            self.cells *= (inputsize[each])
    
    def __mul__(self, other : np.ndarray) -> np.ndarray:
        ls = list(other.shape)

        expression = list(self.inputsize)

        for each in range(len(ls)):
            if ls[each : each + len(self.inputsize)] == expression:
                ls = ls[:each] + [self.cells] + ls[each + len(self.inputsize):]
        
        newshape = tuple(ls)

        return other.reshape(newshape).transpose()

index = 0

class NeuralNetwork:

    def __init__(self, layers : list = None, name : str = '') -> None:
        self.layers = layers if layers is not None else []
        self.name = name if name else f'nn{index}'
        self.cache = ''
        self.record = []
        self.deltas = []
    
    def add(self, layer : Layer) -> None:
        if layer.inputsize is not None:
            assert layer.inputsize == self.layers[-1].cells, f"Invalid layer size; last layer has {self.layers[-1].cells} cells, but input size is {layer.inputsize}"
        self.layers.append(layer)
    
    def compile(self) -> str:
        out = f"{self.name}:\n\n"

        cells = 0

        assert self.layers[0].inputsize is not None, "Cannot infer the input size of the first layer!"

        for each in range(len(self.layers) - 1):
            if self.layers[each + 1].inputsize is not None:
                assert self.layers[each].cells == self.layers[each + 1].inputsize, f"Layer sizes mismatch at {each} and {each + 1}"
            
            else:
                self.layers[each + 1].inputsize = self.layers[each].cells
                self.layers[each + 1].spawn()
            
            try:
                shape = self.layers[each].array.shape
                cells += (shape[0] * shape[1])
            except AttributeError : pass
        
        try:
            shape = self.layers[-1].array.shape
            cells += (shape[0] * shape[1])
        except AttributeError : pass

        out += f"No. of layers : {len(self.layers)}\n"
        out += f"No. of training parameters : {cells}\n\n"

        for each in self.layers:
            # out += f"      |\n"
            out += f"      ↓{each.inputsize}\n"
            out += f"{each.name} ({each.cells} cells)\n"
            # out += f"      |\n"
            out += f"      ↓{each.cells}\n"
        
        out += "\n\n"
        
        self.cache = out

        self.deltas = [None for i in range(len(self.layers))]

        return out
    
    def __repr__(self) -> str:
        return self.cache
    
    def predict(self, vector : np.ndarray) -> np.ndarray:

        self.record.append(vector.copy())

        for layer in self.layers:
            vector = layer * vector
            self.record.append(vector.copy())
            print(f"Recording shape: {vector.copy().shape}")
            vector = layer.activation(vector)
        
        return vector
    
    def fit(self, x : np.ndarray, y : np.ndarray) -> None:
        self.record = []

        self.deltas[-1] = self.predict(x) - y.transpose()

        for each in range(len(self.deltas) - 1):
            each = len(self.deltas) - 2 - each
            # print((self.layers[each + 1].array.shape, self.deltas[each + 1].array.shape))
            # print((self.layers[each + 1].array.shape, self.deltas[each + 1].shape))
            print(f"Reading shape : {self.record[each].shape}")
            delta = np.matmul(self.layers[each + 1].array.transpose(), self.deltas[each + 1])

            gdashz = derivatives[self.layers[each].activation](self.record[each])
            gdashz = np.concatenate((gdashz, [[1 for i in range((gdashz.shape[1]))]]), axis=0)

            delta = delta * gdashz

            self.deltas[each] = delta
    
    def cost(self, X : np.ndarray, Y : np.ndarray) -> float:
        yhat = self.predict(X)

        error = yhat - Y.transpose()

        error = np.linalg.norm(error, axis = 0).reshape((1, X.shape[0]))

        error = np.sum(error)/X.shape[0]

        return error
    