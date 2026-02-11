from models.base_model import BaseModel

class IdentityModel(BaseModel):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return super().forward(x)
    
    def get_representation(self, x, level: int = -1):
        return x.view(x.size(0), -1)
    
    def available_levels(self):
        return [1]
