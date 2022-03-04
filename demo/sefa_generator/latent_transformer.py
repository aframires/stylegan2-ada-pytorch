from torch import load
class LatentTransformer():
    def __init__(self, factorisation_file_path, degree_scaler:int = 5):
        self.eigvecs = load(factorisation_file_path)["eigvec"]
        self.degree_scaler = degree_scaler

    def transform(self, z, transform_weights):
        for idx, degree in enumerate(transform_weights):
            dim_direction = self.__direction(idx, degree)
            z = z + dim_direction

        return z 
        
    def __direction(self, dim_idx, degree):
        current_eigvec = self.eigvecs[:, dim_idx].unsqueeze(0)
        direction = self.degree_scaler * degree * current_eigvec
        return direction

    def transform_shape(self):
        return self.eigvecs.shape