import sys, os
from .utils import here
from pathlib import Path
import threading
import traceback
import warnings
import importlib
from .log import log, blue_text, cyan_text, get_summary, get_label

#Ref: https://github.com/comfyanonymous/ComfyUI/blob/76d53c4622fc06372975ed2a43ad345935b8a551/nodes.py#L17
sys.path.insert(0, str(Path(here, "src").resolve()))
for pkg_name in os.listdir(str(Path(here, "src"))):
    sys.path.insert(0, str(Path(here, "src", pkg_name).resolve()))
print(f"Registered sys.path: {sys.path}")

def load_nodes():
    shorted_errors = []
    full_error_messages = []
    node_class_mappings = {}
    node_display_name_mappings = {}

    for filename in (here / "node_wrappers").iterdir():
        
        module_name = filename.stem
        try:
            module = importlib.import_module(
                f".node_wrappers.{module_name}", package=__package__
            )
            node_class_mappings.update(getattr(module, "NODE_CLASS_MAPPINGS"))
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                node_display_name_mappings.update(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS"))

            log.debug(f"Imported {module_name} nodes")

        except AttributeError:
            pass  # wip nodes
        except Exception:
            error_message = traceback.format_exc()
            full_error_messages.append(error_message)
            error_message = error_message.splitlines()[-1]
            shorted_errors.append(
                f"Failed to import module {module_name} because {error_message}"
            )
    
    if len(shorted_errors) > 0:
        full_err_log = '\n\n'.join(full_error_messages)
        print(f"\n\nFull error log from comfyui_controlnet_aux: \n{full_err_log}\n\n")
        log.info(
            f"Some nodes failed to load:\n\t"
            + "\n\t".join(shorted_errors)
            + "\n\n"
            + "Check that you properly installed the dependencies.\n"
            + "If you think this is a bug, please report it on the github page (https://github.com/Fannovel16/comfyui_controlnet_aux/issues)"
        )
    return node_class_mappings, node_display_name_mappings

AUX_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = load_nodes()

AIO_NOT_SUPPORTED = ["InpaintPreprocessor"]
#For nodes not mapping image to image

class AIO_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        auxs = list(AUX_NODE_MAPPINGS.keys())
        for name in AIO_NOT_SUPPORTED:
            if name in auxs: auxs.remove(name)
        
        return {
            "required": { 
                "image": ("IMAGE",),
                "preprocessor": (auxs, {"default": "CannyEdgePreprocessor"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors"

    def execute(self, preprocessor, image):
        aux_class = AUX_NODE_MAPPINGS[preprocessor]
        input_types = aux_class.INPUT_TYPES()
        input_types = {
            **input_types["required"], 
            **(input_types["optional"] if "optional" in input_types else {})
        }
        params = {}
        for name, input_type in input_types.items():
            if name == "image":
                params[name] = image
                continue
            
            if len(input_type) == 2 and ("default" in input_type[1]):
                params[name] = input_type[1]["default"]
                continue

            default_values = { "INT": 0, "FLOAT": 0.0 }
            if input_type[0] in default_values:
                params[name] = default_values[input_type[0]]

        return getattr(aux_class(), aux_class.FUNCTION)(**params)

NODE_CLASS_MAPPINGS = {
    **AUX_NODE_MAPPINGS,
    "AIO_Preprocessor": AIO_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS.update({
    "AIO_Preprocessor": "AIO Aux Preprocessor"
})
