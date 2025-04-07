import onnxruntime
import numpy as np

def change_onnx_batch_size(model, new_batch_size):
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = new_batch_size
    model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = new_batch_size
    return model

def test_onnx(model, test_set, test_loader, batch_size):
    ort_session = onnxruntime.InferenceSession(model.SerializeToString())

    total_correct = 0
    for indic in test_loader:
        img = indic["img"].numpy()
        lbl = indic["label"].numpy()

        if img.shape[0] != batch_size:
            # Change the batch size of the model
            model = change_onnx_batch_size(model, img.shape[0])
            # Recreate the ort session with the new batch size
            ort_session = onnxruntime.InferenceSession(model.SerializeToString())

        ort_inputs = { 
            ort_session.get_inputs()[0].name: img
        }
        ort_outs = ort_session.run(None, ort_inputs)
        preds = np.argmax(ort_outs[0], axis=1)
        total_correct += np.sum(preds == lbl)

    accuracy = total_correct / len(test_set)
    print("Accuracy: ", accuracy)
