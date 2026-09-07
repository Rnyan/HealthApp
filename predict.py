import pandas as pd
import tensorflow as tf
import joblib as jb

# Load the trained model
model = tf.keras.models.load_model("models/recipe_model.keras")

# reads recipes.csv and restrictions.csv into pandas dataframes
recipes = pd.read_csv("recipes.csv")
restrictions = pd.read_csv("restrictions.csv")

# merges the recipes and restrictions dataframes on recipeID
data = pd.merge(
    recipes,
    restrictions,
    on="recipeID",
    how="left",
    validate="one_to_one",
)

# temporary user requirements for testing
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

# loads the label encoder and scaler from training
label_encoder = jb.load("models/label_encoder.pkl")
scaler = jb.load("models/scaler.pkl")

# filters out recipes that do not meet user's restrictions
allowedRecipes = data[
    (data["vegetarian"] >= userRequirements["vegetarian"].values[0]) &
    (data["vegan"] >= userRequirements["vegan"].values[0]) &
    (data["glutenFree"] >= userRequirements["glutenFree"].values[0]) &
    (data["dairyFree"] >= userRequirements["dairyFree"].values[0]) &
    (data["nutFree"] >= userRequirements["nutFree"].values[0]) &
    (data["lowSugar"] >= userRequirements["lowSugar"].values[0])
]

# If there are no recipes for the user, print it. 
# If there are, predict which one is the best for the user.
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

    # scales the allowed recipes
    allowedScaled = scaler.transform(allowedRecipes[features])
    # predicts the best recipe for the user based on the scaled features
    predictions = model.predict(allowedScaled, verbose=0)
    recipeScores = predictions.max(axis=1)
    # gets the index of the best recipe based on the highest score
    bestRecipeIndex = recipeScores.argmax()
    bestRecipe = allowedRecipes.iloc[bestRecipeIndex]
    # gets the predicted category for the best recipe
    categoryIndex = predictions[bestRecipeIndex].argmax()
    category = label_encoder.inverse_transform([categoryIndex])[0]

    # prints the recommended recipe ID, predicted category, and confidence score
    print(f"Recommended recipe ID: {bestRecipe['recipeID']}")
    print(f"Predicted category: {category}")
    print(f"Confidence: {recipeScores[bestRecipeIndex]:.2%}")