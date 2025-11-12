
from sdks.novavision.src.helper.package import PackageHelper
from capsules.Package.src.models.PackageModel import PackageModel, ConfigExecutor, PackageConfigs, OutputDetections, ForegroundDetectionOutputs, ForegroundDetectionResponse, ForegroundDetectionExecutor, OutputImage


def build_response(context):
    outputImage = OutputImage(value=context.image)
    outputDetections = OutputDetections(value=context.detections)
    Outputs = ForegroundDetectionOutputs(outputImage=outputImage, outputDetections=outputDetections)
    foregroundDetectionResponse = ForegroundDetectionResponse(outputs=Outputs)
    foregroundDetectionExecutor = ForegroundDetectionExecutor(value=foregroundDetectionResponse)
    executor = ConfigExecutor(value=foregroundDetectionExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel