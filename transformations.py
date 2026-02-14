import imgaug.augmenters as iaa
from PIL import Image
import numpy as np
import albumentations as A
import os

def get_transformation(transformation_obj):
    transformation_name = transformation_obj["id"]
    # Create the base transformation
    if transformation_name == "blur":
        sigma_min = transformation_obj.get("sigma_min")
        sigma_max = transformation_obj.get("sigma_max")
        base_transform = iaa.GaussianBlur(sigma=(sigma_min, sigma_max))
    elif transformation_name == "noise":
        scale = transformation_obj.get("scale")
        per_channel = transformation_obj.get("per_channel")
        base_transform = iaa.AdditiveGaussianNoise(scale=scale, per_channel=per_channel)
    elif transformation_name == "jigsaw":
        nb_rows_min = transformation_obj.get("nb_rows_min")
        nb_cols_min = transformation_obj.get("nb_cols_min")
        nb_rows_max = transformation_obj.get("nb_rows_max")
        nb_cols_max = transformation_obj.get("nb_cols_max")
        max_steps_min = transformation_obj.get("max_steps_min")
        max_steps_max = transformation_obj.get("max_steps_max")
        base_transform = iaa.Jigsaw(nb_rows=(nb_rows_min, nb_rows_max), nb_cols=(nb_cols_min, nb_cols_max), max_steps=(max_steps_min, max_steps_max))
    elif transformation_name == "kmeanscolorquantization":
        n_colors_min = transformation_obj.get("n_colors_min")
        n_colors_max = transformation_obj.get("n_colors_max")
        base_transform = iaa.KMeansColorQuantization(n_colors=(n_colors_min, n_colors_max))
    elif transformation_name == "saltandpepper":
        p_min = transformation_obj.get("p_min")
        p_max = transformation_obj.get("p_max")
        per_channel = transformation_obj.get("per_channel")
        base_transform = iaa.SaltAndPepper(p=(p_min, p_max), per_channel=per_channel)
    elif transformation_name == "coarsedropout":
        p_min = transformation_obj.get("p_min")
        p_max = transformation_obj.get("p_max")
        size_percent_min = transformation_obj.get("size_percent_min")
        size_percent_max = transformation_obj.get("size_percent_max")
        per_channel = transformation_obj.get("per_channel")
        base_transform = iaa.CoarseDropout(p=(p_min, p_max), size_percent=(size_percent_min, size_percent_max), per_channel=per_channel)
    elif transformation_name == "multiplyhue":
        hue_min = transformation_obj.get("hue_min")
        hue_max = transformation_obj.get("hue_max")
        base_transform = iaa.MultiplyHue((hue_min, hue_max))
    elif transformation_name == "affine":
        scale_min = transformation_obj.get("scale_min")
        scale_max = transformation_obj.get("scale_max")
        rotate_min = transformation_obj.get("rotate_min")
        rotate_max = transformation_obj.get("rotate_max")
        shear_min = transformation_obj.get("shear_min")
        shear_max = transformation_obj.get("shear_max")
        base_transform = iaa.Affine(scale=(scale_min, scale_max), rotate=(rotate_min, rotate_max), shear=(shear_min, shear_max))
    elif transformation_name == "motionblur":
        kernel_size = transformation_obj.get("kernelsize")
        angle = transformation_obj.get("angle")
        direction = transformation_obj.get("direction")
        base_transform = iaa.MotionBlur(k=kernel_size, angle=angle, direction=direction)
    elif transformation_name == "zoomblur":
        min_factor = transformation_obj.get("minfactor")
        max_factor = transformation_obj.get("maxfactor")
        min_step_factor = transformation_obj.get("minstepfactor")
        max_step_factor = transformation_obj.get("maxstepfactor")
        base_transform = A.ZoomBlur(max_factor=(min_factor, max_factor), step_factor=(min_step_factor, max_step_factor))
    elif transformation_name == "linearcontrast":
        gain = transformation_obj.get("alpha")
        per_channel = transformation_obj.get("perchannel")
        base_transform = iaa.LinearContrast(alpha=gain, per_channel=per_channel)
    elif transformation_name == "grayscale":
        alpha = transformation_obj.get("alpha")
        base_transform = iaa.Grayscale(alpha=alpha)
    elif transformation_name == "fog":
        alpha = transformation_obj.get("alpha")
        min_fog_range = transformation_obj.get("minfogrange")
        max_fog_range = transformation_obj.get("maxfogrange")
        base_transform = A.RandomFog(alpha_coef= alpha, fog_coef_range=(min_fog_range, max_fog_range))
    elif transformation_name == "rain":
        min_slant_range = transformation_obj.get("minslantrange")
        max_slant_range = transformation_obj.get("maxslantrange")
        brightness_coefficient = transformation_obj.get("brightness")
        rain_type = transformation_obj.get("raintype")
        base_transform = A.RandomRain(slant_range=(min_slant_range, max_slant_range), 
                                      brightness_coefficient=brightness_coefficient, 
                                      rain_type = rain_type)
    elif transformation_name == "chromaticabberation":
        min_primary_distortion = transformation_obj.get("minprimarydistortion")
        max_primary_distortion = transformation_obj.get("maxprimarydistortion")
        min_secondary_distortion = transformation_obj.get("minsecondarydistortion")
        max_secondary_distortion = transformation_obj.get("minsecondarydistortion")
        base_transform = A.ChromaticAberration(
            primary_distortion_limit=(min_primary_distortion, max_primary_distortion), 
            secondary_distortion_limit=(min_secondary_distortion, max_secondary_distortion)
        )
    else:
        raise Exception(f"Transformation {transformation_name} is not supported yet!")
    return _wrap_transformation(base_transform)

def _wrap_transformation(transformation):
    def _wrapped_transform(images):
        if type(images)!= list:
            raise Exception("The transformation functions expect a list of images as an input.")
        for img in images:
            if type(img)!= np.ndarray:
                raise Exception("The transformation functions expect images to be represented as numpy ndarrays.")
            if np.any(img < 0) or np.any(img > 1):
                raise ValueError("The transformation functions expect images be scaled between 0 and 1.")
        images_uint8 = [(img * 255).astype(np.uint8) for img in images]
        transformed_images = transformation(images=images_uint8)
        if isinstance(transformed_images, dict) and "images" in transformed_images:
            transformed_images = transformed_images["images"]
        transformed_images_float = [img.astype(np.float32) / 255.0 for img in transformed_images]
        return transformed_images_float
    return _wrapped_transform

def _test_transformations(save = False, save_path = None):
    transformations = [
        {"id": "blur", "sigma_min": 0.1, "sigma_max": 0.5},
        {"id": "noise", "scale": 0.05, "per_channel": True},
        {"id": "jigsaw", "nb_rows_min": 2, "nb_cols_min": 2, "nb_rows_max": 4, "nb_cols_max": 4, "max_steps_min": 10, "max_steps_max": 20},
        {"id": "kmeanscolorquantization", "n_colors_min": 2, "n_colors_max": 4},
        {"id": "saltandpepper", "p_min": 0.01, "p_max": 0.05, "per_channel": True},
        {"id": "coarsedropout", "p_min": 0.01, "p_max": 0.05, "size_percent_min": 0.01, "size_percent_max": 0.05, "per_channel": True},
        {"id": "multiplyhue", "hue_min": -0.1, "hue_max": 0.1},
        {"id": "affine", "scale_min": 0.9, "scale_max": 1.1, "rotate_min": -10, "rotate_max": 10, "shear_min": -10, "shear_max": 10},
        {"id": "motionblur", "kernelsize": 30, "angle": 260, "direction": -1},
        {"id": "zoomblur", "minfactor": 1.0, "maxfactor": 1.13, "minstepfactor": 0.02, "maxstepfactor": 0.03},
        {"id": "linearcontrast", "alpha":0.3, "perchannel": True},
        {"id": "grayscale", "alpha" : 1},
        {"id": "fog", "alpha": 0.08, "minfogrange": 0.3, "maxfogrange": 1},
        {"id": "rain", "minslantrange": -10.0, "maxslantrange": 10.0, "brightness": 0.7, "raintype": "heavy"},
        {"id": "chromaticabberation", "minprimarydistortion": -3.0, "maxprimarydistortion": 3.0, "minsecondarydistortion": -5.0, "maxsecondarydistortion": 5.0}
    ]
    base_iamge = Image.open("luna.jpg")
    base_iamge = np.array(base_iamge)
    base_iamge = base_iamge / 255.0
    transformed_images = []
    for transformation in transformations:
        transformed_image = get_transformation(transformation)([base_iamge])[0]
        transformed_images.append(transformed_images)
        if save:
            transformed_image = np.array(transformed_image) *255
            img = Image.fromarray(transformed_image.astype(np.uint8), "RGB")
            img.save(os.path.join(save_path, f"luna_{transformation['id']}.jpg"))
    return transformed_images


