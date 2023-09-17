from ..utils import common_annotator_call, annotator_ckpts_path, RAFT_MODEL_NAME, create_node_input_types
import comfy.model_management as model_management
import torch

class RaftOpticalFlowPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            model_filename=([
                "raft_large_C_T_SKHT_K_V2-b5c70766.pth", 
                "raft_large_C_T_SKHT_V2-ff5fadd5.pth", 
                "raft_large_C_T_V2-1bb1363a.pth", 
                "raft_small_C_T_V2-01064c6d.pth"
            ], {"default": "raft_large_C_T_SKHT_V2-ff5fadd5.pth"})
        )

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("SIX_CHANNEL_IMAGE", "OPTICAL_FLOW_PREVIEW")
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Line Extractors"

    def execute(self, image, model_filename, resolution=512, **kwargs):
        from controlnet_aux.raft import RaftOpticalFlowEmbedder

        model = RaftOpticalFlowEmbedder.from_pretrained(RAFT_MODEL_NAME, model_filename, cache_dir=annotator_ckpts_path).to(model_management.get_torch_device())
        six_channel_images = common_annotator_call(model, image, input_batch=True, resolution=resolution)
        del model
        #Image tensor of a node is NHWC
        #https://huggingface.co/CiaraRowles/TemporalNet2/blob/main/temporalvideo.py#L264
        return (six_channel_images, six_channel_images[:, :, :, 3:])

NODE_CLASS_MAPPINGS = {
    "RaftOpticalFlowPreprocessor": RaftOpticalFlowPreprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RaftOpticalFlowPreprocessor": "RAFT Optical Flow Embedder (for TemporalNet2)"
}