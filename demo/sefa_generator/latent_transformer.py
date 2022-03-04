from numpy import expand_dims
from torch import load
class LatentTransformer():
    def __init__(self, factorisation_file_path, degree_scaler:int = 1):
        self.eigvecs = load(factorisation_file_path)["eigvec"]
        self.degree_scaler = degree_scaler

    def transform(self, z, transform_weights):
        for idx, degree in enumerate(transform_weights):
            z = z + self.__direction(idx, degree)

        return z
        
    def __direction(self, dim_idx, degree):
        current_eigvec = expand_dims(self.eigvecs[:, dim_idx], 0)
        return self.degree_scaler * degree * current_eigvec

    def transform_shape(self):
        return self.eigvecs.shape