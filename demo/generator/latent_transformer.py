from numpy import expand_dims
from torch import load
from enum import Enum

class TransformerClass(Enum):
    SEFA = 0
    INTERFACE = 1


class SefaTransformer():
    def __init__(self, factorisation_file_path, degree_scaler):
        self.eigvecs = load(factorisation_file_path)["eigvec"]
        self.degree_scaler = degree_scaler

    def transform(self, latent_data, transform_weights):
        for idx, degree in enumerate(transform_weights):
            latent_data += self.__direction(idx, degree)

        return latent_data

    def __direction(self, dim_idx, degree):
        current_eigvec = expand_dims(self.eigvecs[:, dim_idx], 0)
        return self.degree_scaler * degree * current_eigvec


class InterfaceTransformer():
    def __init__(self, factorisation_file_path, degree_scaler):
        self.boundaries = load(factorisation_file_path)
        self.degree_scaler = degree_scaler

    def transform(self, latent_data, transform_weights):
        for boundary, degree in zip(list(self.boundaries.values()), transform_weights):
            latent_data += self.__direction(boundary, degree)

        return latent_data

    def __direction(self, boundary, degree):
        return self.degree_scaler * degree * boundary


class LatentTransformer():
    def __init__(self, tclass: TransformerClass, factorisation_file_path, degree_scaler: int=3):
        if tclass == TransformerClass.SEFA:
            self.transformer = SefaTransformer(factorisation_file_path, degree_scaler)
        elif tclass == TransformerClass.INTERFACE:        
            self.transformer = InterfaceTransformer(factorisation_file_path, degree_scaler)

    def transform(self, latent_data, transform_weights):
        return self.transformer.transform(latent_data, transform_weights)
