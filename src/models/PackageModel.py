
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

class MOGDetectShadows(Config):
    name: Literal["detectShadows"] = "detectShadows"
    value: Union[ConfigTrue, ConfigFalse]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"
    class Config:
        title = "Detect Shadows"

class KNN(Config):
    dist2Threshold: KNNDist2Threshold
    name: Literal["KNN"] = "KNN"
    value: Literal["KNN"] = "KNN"
    type: Literal["string"] = "string"
    field: Literal["Option"] = "Option"
    class Config:
        title = "K-Nearest Neighbors"
class MOG2(Config):
    varThreshold: MOGVarThreshold
    name: Literal["MOG2"] = "MOG2"
    value: Literal["MOG2"] = "MOG2"
    type: Literal["string"] = "string"
    field: Literal["Option"] = "Option"
    class Config:
        title = "Adaptive Mixture of Gaussians v2"

class Type(Config):
    name: Literal["type"] = "type"
    value: Union[MOG2, KNN]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

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
