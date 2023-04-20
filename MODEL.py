# // Importing libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import pickle

# // Loading the Dataset
df = pd.read_csv("data.csv")
print(df.head())

# // EDA - Removing unwanted attributes
# // df.drop('Unnamed: 32',axis=1,inplace=True)
# // df.drop('id',axis=1,inplace=True)

print(df.info())

df['diagnosis']= df['diagnosis'].replace({'M':'Malignant(Cancerous)', 'B':'Benign(Non-Cancerous)'})

# Selecting Independent and Dependent variables
X = df[["radius_mean", "texture_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "area_worst", "smoothness_worst", 'compactness_worst', "concavity_worst", "symmetry_worst", "fractal_dimension_worst"]]
y = df["diagnosis"]

# Train-Test_Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Scalar Transformation
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Instantiate the model
classifier = AdaBoostClassifier()

# Fit the model
classifier.fit(X_train,y_train)

# Make pickle file for the model
pickle.dump(classifier, open("model.pkl", "wb"))