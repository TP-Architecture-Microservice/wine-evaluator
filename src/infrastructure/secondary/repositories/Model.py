from dataclasses import dataclass

from src.domain.model.ModelRepostory import ModelRepository
from src.domain.model.ia.BaseAIModel import BaseAIModel
from src.domain.prediction.Wine import WineWithQuality


@dataclass(frozen=True)
class CSVModelRepository(ModelRepository):
    def persist(self, model: BaseAIModel):
        pass

    def add_entry(self, wine_with_quality: WineWithQuality):
        pass

    def read_csv(self):
        wine_data = pd.read_csv('../../../../Wines.csv')
        wine_data = wine_data.drop('Unnamed: 13', axis=1)
        wine_data = wine_data.drop('Id', axis=1)
        return wine_data
