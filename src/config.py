EVALUATION_METRICS = {
            "AUC-Judd": True,
            "AUC-Borji": True,
            "AUC-Shuff": True, # TODO fix - in alzheimer dataset, gt maps are of size 1920x1080 which is incompatible with the force reduced size to 640x360
            "NSS": True,
            "IG": True,
            "SIM": True,
            "CC": True,
            "KLdiv": True,
}

#GENERALIZED_MAPS_PATH = 'eyetrackingdata/alzheimer-dataset/fixation_map_30_release/Generated_Generalized_maps'
# GENERALIZED_MAPS_PATH = 'eyetrackingdata/alzheimer-dataset-2/Generated_Generalized_maps'
GENERALIZED_MAPS_PATH = 'data/PSD/generated_generalized'
