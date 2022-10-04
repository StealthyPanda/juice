from juice import *

# nn = NeuralNetwork(
#     name = 'Jarvis',
#     layers=[
#         Flatten(inputsize = (28, 28)),
#         Layer(16, activation = actrelu),
#         Layer(16, activation = actrelu),
#         Layer(10, activation = actsigmoid)
#     ]
# )

# nn.compile()

# print(nn)

# v = np.ones((28, 28))

# print(nn.predict(v))

# v = np.random.random(size = (4, 5))
# print(v)
# print(np.linalg.norm(v, axis = 0).reshape(1, 5))

# print(tuple([i for i in range(10)]))

# tup = (1, 2, 5, 6, 4)

# bruh = list(tup)

# ex = [2, 5, 6]
# re = [10, 11, 12, 13]

# for each in range(len(bruh)):
#     if bruh[each:each + len(ex)] == ex:
#         bruh = bruh[: each] + re + bruh[each + len(ex):]

# print(bruh)

v = np.zeros(shape=(28, 28))

f = Flatten((28, 28))

newv = f * v

print(newv.shape)