from dataclasses import dataclass
from uuid import UUID

WineID = UUID

FixedAcidity = float
VolatileAcidity = float
CitricAcid = float
ResidualSugar = float
Chlorides = float
FreeSulfurDioxide = float
TotalSulfurDioxide = float
Density = float
pH = float
Sulphates = float
Alcohol = float

WineQuality = float


@dataclass(frozen=True)
class Wine:
    id: WineID
    fixed_acidity: FixedAcidity
    volatile_acidity: VolatileAcidity
    citric_acid: CitricAcid
    residual_sugar: ResidualSugar
    chlorides: Chlorides
    free_sulfur_dioxide: FreeSulfurDioxide
    total_sulfur_dioxide: TotalSulfurDioxide
    density: Density
    pH: pH
    sulphates: Sulphates
    alcohol: Alcohol


@dataclass(frozen=True)
class WineWithQuality:
    wine: Wine
    quality: WineQuality


