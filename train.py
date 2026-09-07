import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import tensorflow as tf
import joblib as jb
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder, StandardScaler  

tf.get_logger().setLevel("ERROR")
os.makedirs("models", exist_ok=True)

dataC = pd.read_csv("recipes.csv")
dataR = pd.read_csv("restrictions.csv")

data = pd.merge(dataC, dataR, on="recipeID", how="left", validate="one_to_one")

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

x = data[features].values

scaler = StandardScaler()
x = scaler.fit_transform(x)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["category"])

jb.dump(label_encoder, "models/label_encoder.pkl")
jb.dump(scaler, "models/scaler.pkl")

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(len(features),)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    x_train, y_train,
    epochs=100,
    validation_data=(x_test, y_test),
    verbose=1
)

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {accuracy:.2%}")     


model.save("models/recipe_model.keras")

print("Categories: ", list(label_encoder.classes_))
