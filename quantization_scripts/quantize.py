from neural_compressor import quantization, PostTrainingQuantConfig
from neural_compressor.config import AccuracyCriterion
from neural_compressor.data import DataLoader
from neural_compressor.utils.constant import FP32
import dataset_utils

def quantize_onnx_model(
    modelpath,
    test_set,
    approach = "static",
    batch_size = 1,
    quant_level = 1,
    quant_format = "QOperator",
):
    tuple_dloader = dataset_utils.DataloaderTuple(test_set, batch_size=batch_size)

    accuracy_criterion = AccuracyCriterion()
    accuracy_criterion.relative = 0.02

    config = PostTrainingQuantConfig(
        approach= approach,
        quant_level = quant_level,
        quant_format = quant_format,
    )

    q_model = quantization.fit(
        model=modelpath,
        conf=config,
        calib_dataloader=tuple_dloader,
        accuracy_criterion=accuracy_criterion,
    )

    return q_model