from src.domain.model.ia.BaseAIModel import BaseAIModel
from src.domain.model.ia.ModelDisplay import SerializedAIModel, IAModelDescription


class ModelReader:
    @staticmethod
    def get_serialized_model(model: BaseAIModel) -> SerializedAIModel:
        return model.serialize()

    @staticmethod
    def get_model_description(model: BaseAIModel) -> IAModelDescription:
        return model.describe()
