
from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Image, Inputs, Configs, Detection, Outputs, Response, Request, Output, Input, Config


class InputImage(Input):
    """
    Represents the input image(s) for the foreground detection capsule.
    Can accept a single Image or a list of Images.
    """
    name: Literal["inputImage"] = "inputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"
        json_schema_extra = {
            "shortDescription": "Input Image(s)"
        }


class OutputImage(Output):
    """
      Represents the output image(s) from the foreground detection capsule.
      Returns the processed image(s) after foreground detection.
      """
    name: Literal["outputImage"] = "outputImage"
    value: Union[List[Image],Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"
        json_schema_extra = {
            "shortDescription": "Foreground Image(s)"
        }



class OutputDetections(Output):
    """
    Represents the list of detected objects in the processed image(s).
    Each detection contains bounding boxes and metadata.
    """
    name: Literal["outputDetections"] = "outputDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Detections"
        json_schema_extra = {
            "shortDescription": "Detected Objects"
        }



class ConfigTrue(Config):
    """
       Represents a boolean 'True' option in dropdown configurations.
       """
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"
        json_schema_extra = {
            "shortDescription": "Enable Option"
        }


class ConfigFalse(Config):
    """
       Represents a boolean 'False' option in dropdown configurations.
       """
    name: Literal["False"] = "False"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"
        json_schema_extra = {
            "shortDescription": "Disable Option"
        }

class LearningRate(Config):
    """
    Controls the adaptation speed of the background model to changes in the scene.
    Lower values make the model adapt slowly, which helps prevent false positives
    in dynamic environments, but may miss quick changes. Higher values make the model
    adapt faster to new pixel values, allowing it to quickly follow scene changes,
    but may introduce instability or false foreground detections.
    """
    name: Literal["learningRate"] = "learningRate"
    value: float = Field(default=0.03, ge=0.00001, le=1)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    class Config:
        title = "Learning Rate"
        json_schema_extra = {
            "shortDescription": "Controls how fast the model adapts to changes"
        }


class MinContourArea(Config):
    """
    Sets the minimum area (in pixels) for detected foreground contours.
    Contours smaller than this threshold are ignored, which helps reduce
    false detections caused by noise, small shadows, or irrelevant movement.
    Increase this value in crowded scenes or when ignoring small objects is desired.
    """
    name: Literal["minContourArea"] = "minContourArea"
    value: int = Field(default=800, ge=10, le=5000)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    class Config:
        title = "Minimum Contour Area (px)"
        json_schema_extra = {
            "shortDescription": "Minimum size of detected objects to be considered foreground"
        }

class MOGHistory(Config):
    """
    Defines the number of previous frames used to build the background model.
    A higher history value makes the model more stable and resistant to noise,
    but slower to react to sudden scene changes. Lower values make the model
    more sensitive to changes but may cause flickering in the foreground mask.
    """
    name: Literal["history"] = "history"
    value: int = Field(default=200, ge=1, le=10000)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    class Config:
        title = "History"
        json_schema_extra = {
            "shortDescription": "Number of past frames considered in background model"
        }

class MOGVarThreshold(Config):
    """
    Sets the variance threshold for the MOG background subtraction.
    This controls how sensitive the model is to changes in pixel values.
    Lower values detect smaller variations but may increase noise,
    while higher values make the model less sensitive to minor changes.
    """
    name: Literal["varThreshold"] = "varThreshold"
    value: float = Field(default=16.0, ge=1.0, le=100.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    class Config:
        title = "Variance Threshold"
        json_schema_extra = {
            "shortDescription": "Sensitivity threshold for pixel variance"
        }

class KNNDist2Threshold(Config):
    """
    Distance threshold for KNN-based background subtraction.
    Determines if a pixel matches existing background samples. Pixels with
    distance above this threshold are considered foreground. Lower values
    make the model stricter (fewer false positives), while higher values
    increase tolerance but may miss subtle changes.
    """
    name: Literal["dist2Threshold"] = "dist2Threshold"
    value: float = Field(default=400.0, ge=1.0, le=5000.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    class Config:
        title = "Distance Threshold"
        json_schema_extra = {
            "shortDescription": "Maximum distance to match background pixel"
        }

class KNNNSamples(Config):
    """
    Number of background samples stored for each pixel in the KNN model.
    More samples improve the robustness of the background model against
    noise and repeated transient movements, but increase memory usage.
    """
    name: Literal["nSamples"] = "nSamples"
    value: int = Field(default=20, ge=1, le=100)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Number of Samples"
        json_schema_extra = {
            "shortDescription": "Number of stored background samples per pixel"
        }


class KNNkNNSamples(Config):
    """
    Number of nearest neighbors used in KNN classification to determine
    if a pixel belongs to the background. A higher number provides more
    robust decisions but may smooth out subtle foreground changes.
    """
    name: Literal["kNNSamples"] = "kNNSamples"
    value: int = Field(default=2, ge=1, le=20)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "KNN Sample Count"
        json_schema_extra = {
            "shortDescription": "Number of neighbors considered in KNN"
        }


class MOGDetectShadows(Config):
    """
    Option to enable or disable shadow detection in background subtraction.
    When enabled, pixels identified as shadows are marked differently from
    the foreground, helping improve object detection accuracy. Useful in
    outdoor environments with varying lighting conditions.
    """
    name: Literal["detectShadows"] = "detectShadows"
    value: Union[ConfigTrue, ConfigFalse]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"
    class Config:
        title = "Detect Shadows"
        json_schema_extra = {
            "shortDescription": "Enable or disable shadow marking in the mask"
        }


class MOG2NMixtures(Config):
    """
    Specifies the number of Gaussian mixtures per pixel in the MOG2 background model.
    More mixtures allow the model to better represent complex background variations,
    but increase computational cost. Typical values are 3–5 for most applications.
    """
    name: Literal["nMixtures"] = "nMixtures"
    value: int = Field(default=3, ge=1, le=10)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Number Gaussian Mixtures"
        json_schema_extra = {
            "shortDescription": "Mixtures per Pixel"
        }


class MOG2ShadowThreshold(Config):
    """
    Defines the threshold for detecting shadows in the MOG2 background model.
    Pixels with intensity changes below this value may be classified as shadows.
    Higher values make the algorithm more sensitive to shadows,
    while lower values reduce shadow detection.
    """
    name: Literal["shadowThreshold"] = "shadowThreshold"
    value: float = Field(default=0.7, ge=0.0, le=1.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Shadow Threshold"
        json_schema_extra = {
            "shortDescription": "Shadow Detection Sensitivity"
        }


class MOG2BackgroundRatio(Config):
    """
    Defines the minimum fraction of history that a pixel must match to be considered background.
    Higher values make background detection stricter, reducing false positives but potentially
    missing slowly moving objects.
    """
    name: Literal["backgroundRatio"] = "backgroundRatio"
    value: float = Field(default=0.8, ge=0.0, le=1.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Background Ratio"
        json_schema_extra = {
            "shortDescription": "Fraction of history to treat pixel as background"
        }


class MOG2VarMin(Config):
    """
       The minimum allowed variance for a Gaussian in the MOG2 background model.
       Pixels with variance below this threshold are treated as stable background.
       """
    name: Literal["varMin"] = "varMin"
    value: float = Field(default=16.0, ge=1.0, le=10000.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Minimum Variance"
        json_schema_extra = {
            "shortDescription": "Minimum allowed variance for stable background pixels"
        }


class MOG2VarThresholdGen(Config):
    """
       Threshold used during variance generation in MOG2.
       Controls how new pixel variations are incorporated into the background model.
       """
    name: Literal["varThresholdGen"] = "varThresholdGen"
    value: float = Field(default=9.0, ge=0.0, le=1000.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Variance Threshold Generation"
        json_schema_extra = {
            "shortDescription": "Threshold for integrating new pixel variations"
        }

class MOG2VarMax(Config):
    """
        Maximum allowed variance for a Gaussian in MOG2 background model.
        Helps to limit extreme fluctuations and prevent false foreground detections.
        """
    name: Literal["varMax"] = "varMax"
    value: float = Field(default=5625.0, ge=1.0, le=20000.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Maximum Variance"
        json_schema_extra = {
            "shortDescription": "Maximum variance allowed to avoid false detections"
        }

class MOG2VarInit(Config):
    """
    Initial variance value assigned to new Gaussians in MOG2.
    Determines initial sensitivity of newly created background components.
    """
    name: Literal["varInit"] = "varInit"
    value: float = Field(default=15.0, ge=0.0, le=10000.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Initial Variance"
        json_schema_extra = {
            "shortDescription": "Starting variance for newly added background Gaussians"
        }


class Threshold(Config):
    """
        Threshold for binarizing the foreground mask.
        Pixels with intensity above this value are considered foreground.
        """
    name: Literal["threshold"] = "threshold"
    value: int = Field(default=30, ge=5, le=255)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Threshold"
        json_schema_extra = {
            "shortDescription": "Intensity threshold for detecting foreground pixels"
        }

class MOG2ComplexityReductionThreshold(Config):
    """
      Threshold controlling complexity reduction in MOG2.
      Used to remove low-weight Gaussians and reduce computational load.
      """
    name: Literal["complexityReductionThreshold"] = "complexityReductionThreshold"
    value: float = Field(default=0.05, ge=0.0, le=1.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Complexity Reduction Threshold"
        json_schema_extra = {
            "shortDescription": "Threshold to remove insignificant Gaussians and reduce computation"
        }

class KNN(Config):
    history: MOGHistory
    detectShadows: MOGDetectShadows
    dist2Threshold: KNNDist2Threshold
    threshold: Threshold
    nSamples: KNNNSamples
    kNNSamples: KNNkNNSamples
    learningRate: LearningRate
    minContourArea: MinContourArea
    name: Literal["KNN"] = "KNN"
    value: Literal["KNN"] = "KNN"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"
    class Config:
        title = "KNN"
        
class MOG2(Config):
    history: MOGHistory
    detectShadows: MOGDetectShadows
    threshold: Threshold
    varThreshold: MOGVarThreshold
    nMixtures: MOG2NMixtures
    shadowThreshold: MOG2ShadowThreshold
    backgroundRatio: MOG2BackgroundRatio
    varMin: MOG2VarMin
    varMax: MOG2VarMax
    varInit: MOG2VarInit
    complexityReductionThreshold: MOG2ComplexityReductionThreshold
    varThresholdGen: MOG2VarThresholdGen
    learningRate: LearningRate
    minContourArea: MinContourArea
    name: Literal["MOG2"] = "MOG2"
    value: Literal["MOG2"] = "MOG2"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"
    class Config:
        title = "MOG2"


class FrameDifferencing(Config):
    """
    Foreground detection method using frame differencing.
    Pixels that differ from the previous frame by more than the threshold are considered foreground.
    """
    threshold: Threshold
    minContourArea: MinContourArea
    name: Literal["FrameDifferencing"] = "FrameDifferencing"
    value:Literal["FrameDifferencing"] = "FrameDifferencing"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Frame Differencing"
        json_schema_extra = {
            "shortDescription": "Detects moving objects by comparing consecutive frames"
        }


class BGInitDuration(Config):
    """
     Duration in seconds to initialize the background model.
     During this period, the model learns the static background before foreground detection begins.
     """
    name: Literal["bgInitDuration"] = "bgInitDuration"
    value: int = Field(default=5, ge=1, le=600)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    class Config:
        title = "BG Init Time (s)"
        json_schema_extra = {
            "shortDescription": "Time for background initialization in seconds"
        }

class BGModeMean(Config):
    """
      Background computation mode using mean values of initial frames.
      """
    name: Literal["Mean"] = "Mean"
    value: Literal["Mean"] = "Mean"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Mean"
        json_schema_extra = {
            "shortDescription": "Compute background using mean of frames"
        }

class BGModeMedian(Config):
    """
     Background computation mode using median values of initial frames.
     """
    name: Literal["Median"] = "Median"
    value: Literal["Median"] = "Median"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Median"
        json_schema_extra = {
            "shortDescription": "Compute background using median of frames"
        }

class BGMode(Config):
    """
     Select background calculation method for running average background model.
     """
    name: Literal["bgMode"] = "bgMode"
    value: Union[BGModeMean, BGModeMedian]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"
    class Config:
        title = "Background Mode"
        json_schema_extra = {
            "shortDescription": "Choose how to compute initial background (Mean or Median)"
        }


class RunningAverage(Config):
    threshold: Threshold
    learningRate: LearningRate
    minContourArea: MinContourArea
    bgInitDuration: BGInitDuration
    bgMode: BGMode
    name: Literal["RunningAverage"] = "RunningAverage"
    value: Literal["RunningAverage"] = "RunningAverage"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Running Average"


class Type(Config):
    """
     Select the foreground detection algorithm to use.
     Options include MOG2, KNN, Frame Differencing, and Running Average.
     """
    name: Literal["type"] = "type"
    value: Union[MOG2, KNN, FrameDifferencing, RunningAverage]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Type"
        json_schema_extra = {
            "shortDescription": "Choose the foreground detection method"
        }

class ForegroundDetectionInputs(Inputs):
    inputImage: InputImage


class ForegroundDetectionConfigs(Configs):
    type: Type


class ForegroundDetectionOutputs(Outputs):
    outputImage: OutputImage
    outputDetections: OutputDetections


class ForegroundDetectionRequest(Request):
    inputs: Optional[ForegroundDetectionInputs]
    configs: ForegroundDetectionConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class ForegroundDetectionResponse(Response):
    outputs: ForegroundDetectionOutputs


class ForegroundDetectionExecutor(Config):
    name: Literal["ForegroundDetection"] = "ForegroundDetection"
    value: Union[ForegroundDetectionRequest, ForegroundDetectionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Foreground Detection"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[ForegroundDetectionExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {
            "target": "value"
        }


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"
    name: Literal["ForegroundDetection"] = "ForegroundDetection"
