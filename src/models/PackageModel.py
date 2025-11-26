
from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Image, Inputs, Configs, Detection, Outputs, Response, Request, Output, Input, Config


class InputImage(Input):
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


class OutputImage(Output):
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


class OutputDetections(Output):
    name: Literal["outputDetections"] = "outputDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Detections"



class ConfigTrue(Config):
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


class ConfigFalse(Config):
    name: Literal["False"] = "False"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class MOGHistory(Config):
    name: Literal["history"] = "history"
    value: int = Field(default=200, ge=1, le=1000)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    class Config:
        title = "History"

class MOGVarThreshold(Config):
    name: Literal["varThreshold"] = "varThreshold"
    value: float = Field(default=16.0, ge=1.0, le=100.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    class Config:
        title = "Variance Threshold"

class KNNDist2Threshold(Config):
    name: Literal["dist2Threshold"] = "dist2Threshold"
    value: float = Field(default=400.0, ge=1.0, le=5000.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    class Config:
        title = "Distance Threshold"

class KNNNSamples(Config):
    name: Literal["nSamples"] = "nSamples"
    value: int = Field(default=20, ge=1, le=100)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Number of Samples"


class KNNkNNSamples(Config):
    name: Literal["kNNSamples"] = "kNNSamples"
    value: int = Field(default=2, ge=1, le=20)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "KNN Sample Count"

class MOGDetectShadows(Config):
    name: Literal["detectShadows"] = "detectShadows"
    value: Union[ConfigTrue, ConfigFalse]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"
    class Config:
        title = "Detect Shadows"


class MOG2NMixtures(Config):
    name: Literal["nMixtures"] = "nMixtures"
    value: int = Field(default=3, ge=1, le=10)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Number Gaussian Mixtures"


class MOG2ShadowThreshold(Config):
    name: Literal["shadowThreshold"] = "shadowThreshold"
    value: float = Field(default=0.7, ge=0.0, le=1.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Shadow Threshold"


class MOG2BackgroundRatio(Config):
    name: Literal["backgroundRatio"] = "backgroundRatio"
    value: float = Field(default=0.8, ge=0.0, le=1.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Background Ratio"


class MOG2VarMin(Config):
    name: Literal["varMin"] = "varMin"
    value: float = Field(default=16.0, ge=1.0, le=10000.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Minimum Variance"


class MOG2VarThresholdGen(Config):
    name: Literal["varThresholdGen"] = "varThresholdGen"
    value: float = Field(default=9.0, ge=0.0, le=1000.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Variance Threshold Generation"
class MOG2VarMax(Config):
    name: Literal["varMax"] = "varMax"
    value: float = Field(default=5625.0, ge=1.0, le=20000.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Maximum Variance"

class MOG2VarInit(Config):
    name: Literal["varInit"] = "varInit"
    value: float = Field(default=15.0, ge=0.0, le=10000.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Initial Variance"


class MOG2ComplexityReductionThreshold(Config):
    name: Literal["complexityReductionThreshold"] = "complexityReductionThreshold"
    value: float = Field(default=0.05, ge=0.0, le=1.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Complexity Reduction Threshold"

class KNN(Config):
    dist2Threshold: KNNDist2Threshold
    nSamples: KNNNSamples
    kNNSamples: KNNkNNSamples
    name: Literal["KNN"] = "KNN"
    value: Literal["KNN"] = "KNN"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"
    class Config:
        title = "KNN"
        
class MOG2(Config):
    varThreshold: MOGVarThreshold
    nMixtures: MOG2NMixtures
    shadowThreshold: MOG2ShadowThreshold
    backgroundRatio: MOG2BackgroundRatio
    varMin: MOG2VarMin
    varMax: MOG2VarMax
    varInit: MOG2VarInit
    complexityReductionThreshold: MOG2ComplexityReductionThreshold
    varThresholdGen: MOG2VarThresholdGen
    name: Literal["MOG2"] = "MOG2"
    value: Literal["MOG2"] = "MOG2"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"
    class Config:
        title = "MOG2"

class Type(Config):
    name: Literal["type"] = "type"
    value: Union[MOG2, KNN]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Type"

class ForegroundDetectionInputs(Inputs):
    inputImage: InputImage


class ForegroundDetectionConfigs(Configs):
    type: Type
    history: MOGHistory
    detectShadows: MOGDetectShadows


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
