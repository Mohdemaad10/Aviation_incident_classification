import joblib
from tensorflow.keras.models import load_model
import numpy as np

class NNclassifier():
    def __init__(self):
        self.vectorizer=joblib.load("vectorizer_object.joblib")
        self.encoder=joblib.load("label_encoder.joblib")
        self.model=load_model("nn_model.keras")

    def predict_text(self,text):
        transformed_text=self.vectorizer.transform([text])
        probabilites=self.model.predict(transformed_text,verbose=0)
        class_index=np.argmax(probabilites,axis=1)
        label=self.encoder.inverse_transform(class_index)

        return label[0] 



