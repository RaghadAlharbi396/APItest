from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model and scaler
try:
    model = joblib.load('Models/knn_model.joblib')
    scaler = joblib.load('Models/scaler.joblib')
except Exception as e:
    logger.error(f"Error loading model or scaler: {str(e)}")
    raise RuntimeError(f"Error loading model or scaler: {str(e)}")

# Define input schema
class InputFeatures(BaseModel):
    Year: int
    Engine_Size: float
    Mileage: float
    Type: str
    Make: str
    Options: str

# Data preprocessing function
def preprocessing(input_features: InputFeatures):
    # Create a dictionary with the required features
    dict_f = {
        'Year': input_features.Year,
        'Engine_Size': input_features.Engine_Size,
        'Mileage': input_features.Mileage,
        'Type_Accent': input_features.Type == 'Accent',
        'Type_Land Cruiser': input_features.Type == 'Land Cruiser',
        'Make_Hyundai': input_features.Make == 'Hyundai',
        'Make_Mercedes': input_features.Make == 'Mercedes',
        'Options_Full': input_features.Options == 'Full',
        'Options_Standard': input_features.Options == 'Standard'
    }

    # Convert dictionary values to a list in the correct order
    features_list = [dict_f[key] for key in sorted(dict_f)]

    # Scale the input features
    scaled_features = scaler.transform([features_list])

    return scaled_features

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to Tuwaiq Academy"}

# POST request for prediction
@app.post("/predict")
async def predict(input_features: InputFeatures):
    data = preprocessing(input_features)
    y_pred = model.predict(data)
    return {"pred": y_pred.tolist()[0]}