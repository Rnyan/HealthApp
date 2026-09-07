import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import tensorflow as tf
import joblib as jb
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder, StandardScaler  

# Suppress TensorFlow logging
tf.get_logger().setLevel("ERROR")
os.makedirs("models", exist_ok=True)

# reads recipes.csv and restrictions.csv into pandas dataframes
dataC = pd.read_csv("recipes.csv")
dataR = pd.read_csv("restrictions.csv")

# merges the recipes and restrictions dataframes based on recipeID
data = pd.merge(dataC, dataR, on="recipeID", how="left", validate="one_to_one")

# defines the features for the model
features = [
    "calories",
    "protein",
    "prepTime",
    "highFiber",
    "vegetarian",
    "vegan",
    "glutenFree",
    "dairyFree",
    "nutFree",
    "lowSugar",
]

# defines the target variable for the model
x = data[features].values
# scales the features using StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)
# encodes the target variable using LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["category"])

# saves the label encoder and scaler for later use in prediction
jb.dump(label_encoder, "models/label_encoder.pkl")
jb.dump(scaler, "models/scaler.pkl")

# splits the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# defines the model's network with Keras Sequential API
model = tf.keras.Sequential([
    tf.keras.Input(shape=(len(features),)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation="softmax")
])

# compiles the model with Adam optimizer and sparse categorical crossentropy loss
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# trains the model on the training data for 5000 epochs with validation on the test data
model.fit(
    x_train, y_train,
    epochs=100,
    validation_data=(x_test, y_test),
    verbose=1
)

# evaluates the model on the test data and prints the accuracy
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {accuracy:.2%}")     

# saves the trained model in Keras format for later use in prediction
model.save("models/recipe_model.keras")
# prints the categories that the model can predict
print("Categories: ", list(label_encoder.classes_))
