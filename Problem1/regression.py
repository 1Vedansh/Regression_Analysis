import numpy as np

# Initialize data lists
data = []

# Initialize target variable lists
countData = []
casualData = []
registeredData = []

# Standard scaling function
def standard_scale(X):
    X = np.array(X)
    
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    
    constant_cols = (std == 0)
    mean[constant_cols] = 0.0
    std[constant_cols] = 1.0
    
    X_scaled = (X - mean) / std
    
    return X_scaled, mean, std

# Calculate Mean Squared Error
def calculateMSE(yTrue, yPred):
    yTrueArray = np.array(yTrue)
    yPredArray = np.array(yPred)

    mse = np.mean((yTrueArray - yPredArray) ** 2)
    return mse

# Calculate Coefficient of Determination (R^2)
def calculateCOD(yTrue, yPred):
    yTrueArray = np.array(yTrue)
    yPredArray = np.array(yPred)

    ssRes = np.sum((yTrueArray - yPredArray) ** 2)
    ssTot = np.sum((yTrueArray - np.mean(yTrueArray)) ** 2)

    r2 = 1 - (ssRes / ssTot)
    return r2

# Generate weights using Regression Equation
def generateWeights(basis, y):
    X = np.array(basis)
    Y = np.array(y)

    XtX = X.T @ X
    
    # pseudo-inverse in case XtX is not invertible
    XtX_inv = np.linalg.pinv(XtX)

    XtY = X.T @ Y

    WstarY = XtX_inv @ XtY
    
    return WstarY

# Basis function generators
# Linear Basis
def generateLinearBasis(dataPoint):
    return [1.0] + dataPoint

# Coupling Quadratic Basis
def generateCouplingQuadraticBasis(dataPoint):
    basis = [1.0]
    for i in range(len(dataPoint)):
        basis.append(dataPoint[i])
    for i in range(len(dataPoint)):
        for j in range(i, len(dataPoint)):
            basis.append(dataPoint[i] * dataPoint[j])
    return basis

# Polynomial Basis (No interaction terms)
def generatePolynomialBasis(dataPoint, degree):
    basis = [1.0]
    n = len(dataPoint)

    for i in range(len(dataPoint)):
        for d in range(1, degree + 1):
            basis.append(dataPoint[i] ** d)
    return basis

# Regression functions
# Linear Regression
def linearRegression(data, y):
    # Basis matrix
    basisList = []

    for i in range(len(data)):
        basisList.append(generateLinearBasis(data[i]))

    scaledBasisList, mean, std = standard_scale(basisList)

    return generateWeights(scaledBasisList, y), mean, std

# Coupling Quadratic Regression
def couplingQuadraticRegression(data, y):
    # Basis matrix
    basisList = []

    for i in range(len(data)):
        basisList.append(generateCouplingQuadraticBasis(data[i]))

    scaledBasisList, mean, std = standard_scale(basisList)

    return generateWeights(scaledBasisList, y), mean, std

# Polynomial Regression
def polynomialRegression(data, y, degree):
    # Basis matrix
    basisList = []

    for i in range(len(data)):
        basisList.append(generatePolynomialBasis(data[i], degree))

    scaledBasisList, mean, std = standard_scale(basisList)

    return generateWeights(scaledBasisList, y), mean, std

# Read partially processed training data
with open(file="processedTrain.csv", mode="r") as file:
    lines = file.readlines()
    for index, line in enumerate(lines):
        if index == 0:
            continue
        
        columns = line.strip().split(",")
        
        # converting the features to appropriate float
        features = [float(col) for col in columns[:-3]]

        # Our target variables
        casualData.append(float(columns[-3]))
        registeredData.append(float(columns[-2]))
        countData.append(float(columns[-1]))

        # append the features to data list
        data.append(features)

# Read partially processed test data and make predictions 
with open(file = "processedTest.csv", mode = "r") as file, open(file = "countPredict.csv", mode = "w") as outputFile:
    lines = file.readlines()

    print("Select Option:\n1. Linear Regression \n2. Coupling Quadratic Regression \n3. Polynomial Regression")

    # Take user input for option
    option = int(input())
    if option == 1:
        weights,mean,std = linearRegression(data, countData)
    elif option == 2:
        weights,mean,std = couplingQuadraticRegression(data, countData)
    elif option == 3:
        print("Enter Degree of Polynomial:")
        degree = int(input())
        weights,mean,std = polynomialRegression(data, countData, degree)

    outputFile.write("realCount,predictedCount\n")

    yTrue = []
    yPred = []

    # Process test data
    for index, line in enumerate(lines):
        if index == 0:
            continue
        
        columns = line.strip().split(",")
        features = [float(col) for col in columns[:-3]]

        # Generate basis according to selected option
        if option == 1:
            features_array = np.array(generateLinearBasis(features))
        elif option == 2:
            features_array = np.array(generateCouplingQuadraticBasis(features))
        elif option == 3:
            features_array = np.array(generatePolynomialBasis(features, degree))

        # apply the same standard scaling as training data (we are not reading the test case here so no leakage)
        features_array = (features_array - mean) / std

        # Make prediction
        prediction = features_array @ weights
        yTrue.append(float(columns[-1]))
        yPred.append(prediction)
        outputFile.write(f"{columns[-1]},{prediction}\n")

    mse = calculateMSE(yTrue, yPred)
    r2 = calculateCOD(yTrue, yPred)
    print(f"MSE on Test Data: {mse}")
    print(f"R^2 on Test Data: {r2}")