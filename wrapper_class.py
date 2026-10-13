import joblib
import numpy as np
import onnxruntime as ort

class NNclassifier():
    def __init__(self):
        self.vectorizer=joblib.load("vectorizer_object.joblib")
        self.encoder=joblib.load("label_encoder.joblib")
        self.session=ort.InferenceSession("NN_model.onnx")

    def predict_text(self,text):
        transformed_text=self.vectorizer.transform([text])
        input_name=self.session.get_inputs()[0].name
        output=self.session.run(None,{input_name: transformed_text.toarray().astype(np.float32)})[0]
        class_index=np.argmax(output,axis=1)
        label=self.encoder.inverse_transform(class_index)

        return label[0] 





