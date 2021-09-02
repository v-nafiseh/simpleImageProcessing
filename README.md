### simple classification

- taken a set of a hundred flower images and arranged them into K categories
- made a feature vector per image and added category of each image at the end of the feature vector
- the dataset is divided into train and test
- the classification is done by supervised learning using RandomForestClassifier
- the result is determined with test data

####brief overview 
- loading data(organizing and loading data)
- Preprocessing(resizing, feature extraction, feature vector)
- Build & Test model(divide dataset into train & test)

######global feature descriptors
- color (color histogram)
- shape (hue moments)
- haralick

######local feature descriptors
- descriptors that evaluate local parts of an image locally 
