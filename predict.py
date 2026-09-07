import pandas as pd
import tensorflow as tf
import joblib as jb

model = tf.keras.models.load_model("models/recipe_model.keras")

recipes = pd.read_csv("recipes.csv")
restrictions = pd.read_csv("restrictions.csv")

data = pd.merge(
    recipes,
    restrictions,
    on="recipeID",
    how="left",
    validate="one_to_one",
)

userRequirements = pd.DataFrame([{
    "calories": 500,
    "protein": 30,
    "prepTime": 20,
    "highFiber": 1,
    "vegetarian": 1,
    "vegan": 0,
    "glutenFree": 1,
    "dairyFree": 1,
    "nutFree": 1,
    "lowSugar": 0,
}])

label_encoder = jb.load("models/label_encoder.pkl")
scaler = jb.load("models/scaler.pkl")

allowedRecipes = data[
    (data["vegetarian"] >= userRequirements["vegetarian"].values[0]) &
    (data["vegan"] >= userRequirements["vegan"].values[0]) &
    (data["glutenFree"] >= userRequirements["glutenFree"].values[0]) &
    (data["dairyFree"] >= userRequirements["dairyFree"].values[0]) &
    (data["nutFree"] >= userRequirements["nutFree"].values[0]) &
    (data["lowSugar"] >= userRequirements["lowSugar"].values[0])
]

if allowedRecipes.empty:
    print("No recipes found matching the dietary restrictions.")
else:
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

    allowedScaled = scaler.transform(allowedRecipes[features])
    predictions = model.predict(allowedScaled, verbose=0)
    recipeScores = predictions.max(axis=1)
    bestRecipeIndex = recipeScores.argmax()
    bestRecipe = allowedRecipes.iloc[bestRecipeIndex]
    categoryIndex = predictions[bestRecipeIndex].argmax()
    category = label_encoder.inverse_transform([categoryIndex])[0]

    print(f"Recommended recipe ID: {bestRecipe['recipeID']}")
    print(f"Predicted category: {category}")
    print(f"Confidence: {recipeScores[bestRecipeIndex]:.2%}")