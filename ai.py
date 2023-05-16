from sklearn import tree

# Define the training data
# Features: [height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42],
     [181, 85, 43]]

# Labels: gender
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# Create and train the decision tree classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)

# Make a prediction
prediction = clf.predict([[190,70,43]])

# Print the prediction
print(prediction) # ['male']