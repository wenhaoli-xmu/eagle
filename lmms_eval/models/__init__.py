import os

# AVAILABLE_MODELS = {
#     "eagle": "Eagle",
# }
# AVAILABLE_MODELS = {
#     "eagle_v1": "Eagle",
# }
AVAILABLE_MODELS = {
     "eagle_v2": "Eagle",
}
# AVAILABLE_MODELS = {
#      "eagle_v3": "Eagle",
# }

for model_name, model_class in AVAILABLE_MODELS.items():
    try:
        exec(f"from .{model_name} import {model_class}")
    except ImportError:
        pass


import hf_transfer

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
