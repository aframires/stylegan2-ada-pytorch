import torch
import numpy as np

boundary_dict = {}
features = ['hardness', 'depth', 'brightness',
            'roughness', 'warmth', 'sharpness', 'boominess']

for feature in features:
    boundary_dict[feature] = torch.from_numpy(np.load('boundaries/' + feature + '/boundary.npy'))

torch.save(boundary_dict,'boundaries.pt')