import pickle
from sklearn.pipeline import Pipeline

def load_pipeline(pipeline_filepath) -> Pipeline :
    """
    Load a machine learning pipeline from a file.
    Args:
        pipeline_filepath (str): The file path to the pipeline file.
    Returns:
        Pipeline: The loaded machine learning pipeline object if successful, 
                    otherwise None if an error occurs during loading.
    """
    try:
        with open(pipeline_filepath, "rb") as file:
            pipeline = pickle.load(file)
        return pipeline
    except:
        return None