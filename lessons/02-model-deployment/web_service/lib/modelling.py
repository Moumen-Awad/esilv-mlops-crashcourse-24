#def (data: list, preprocessor, model):
from scipy.sparse import csr_matrix
from lib.utils import load_model

def run_inference(input_data: csr_matrix, preprocessor, model_path: str):
    x = preprocessor.transform(input_data)
    model = load_model(model_path)
    return model.predict(x)
