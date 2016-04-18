import numpy as np
from random import uniform

class NeuralNet:
	def __init__(self):
		self.w1 = 0.0
		self.w2 = 3.0

	def forward(self, X):
		z2 = self.getZ2(X)

		if (z2 > 1):
			return 1.0
		else:
			return 0.0

	def getZ2(self, X):
		return X[0] * self.w1 + X[1] * self.w2

	def computeGradient(self, X, y):
		r = self.forward(X) - y

		deltaOne = -1 * (r**3) * X[0] - (r**2) * X[0] + r * X[0]
		deltaTwo = -1 * (r**3) * X[1] - (r**2) * X[1] + r * X[1]

		return deltaOne, deltaTwo

def createDataSet(n):
	# The array for inputs
	data = []

	# The array for the expected results
	expected = []

	# Generate n data pieces
	for index in range(n):
		x = uniform(-5, 5)
		y = uniform(-5, 5)

		data.append([x, y])

		expectedValue = 0.0

		if (y > 1):
			expectedValue = 1.0

		expected.append(expectedValue)

	return data, expected

# Construct the neural net object
nn = NeuralNet()

# Create some data
X, y = createDataSet(100000)

# The learning constant
k = 0.001

recordedPositions = []

def train(nn):
	# Probability that the method will go uphill
	prob = 40.0

	# Train via normal method
	for i in range(len(X)):
		dW1, dW2 = nn.computeGradient(X[i], y[i])

		# Scalars that decide if the algorithm will go uphill or downhill
		w1Scalar = 1
		w2Scalar = 1

		if (uniform(0, 100) < prob):
			w1Scalar = -1
		if (uniform(0, 100) < prob):
			w2Scalar = -1

		# Adjust weights
		nn.w1 = nn.w1 - k * dW1 * w1Scalar
		nn.w2 = nn.w2 - k * dW2 * w2Scalar

		if (i % 100 == 0):
			recordedPositions.append([nn.w1, nn.w2])

for a in range(3):
	for b in range(3):
		nn.w1 = -1 + a
		nn.w2 = 0 + b

		train(nn)

		recordedPositions.append([])


# Write results to file
fo = open("marcoPositions.csv", "w")

for i in range(len(recordedPositions)):
	if (len(recordedPositions[i]) == 0):
		fo.write('\n')
	else:
		fo.write(str(recordedPositions[i][0]))
		fo.write(',')
		fo.write(str(recordedPositions[i][1]))
		fo.write('\n')

fo.close()