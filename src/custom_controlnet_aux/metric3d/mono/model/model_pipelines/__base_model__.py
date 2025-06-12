import torch
import torch.nn as nn
from ...utils.comm import get_func


class BaseDepthModel(nn.Module):
    def __init__(self, cfg, **kwargs) -> None:
        super(BaseDepthModel, self).__init__()
        model_type = cfg.model.type
        # Use relative import approach - get the module dynamically
        from . import dense_pipeline
        if model_type == 'DensePredModel':
            self.depth_model = dense_pipeline.DensePredModel(cfg)
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented")

    def forward(self, data):
        output = self.depth_model(**data)

        return output['prediction'], output['confidence'], output

    def inference(self, data):
        with torch.no_grad():
            pred_depth, confidence, _ = self.forward(data)
        return pred_depth, confidence