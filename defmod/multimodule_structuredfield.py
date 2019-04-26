import torch
import defmod.deformationmodules.structuredfield

class StructuredField_multi(StructuredField):
    def __init__(self, fields, points_in_region):
        super().__init__()
        self.__fields = fields
        self.__nb_fields = len(fields)
        self.__points_in_region = points_in_region
        
    @property
    def fields(self):
        return self.__fields

    @property
    def nb_field(self):
        return len(self.__fields)

    def __getitem__(self, index):
        return self.__fields

    def __call__(self, points, k=0):
        multifield = toch.zeros(points.shape)
        for i in range(len(points_in_region)):
            label = np.array(points_in_region[i](points)).astype(int)
            field = fields[i](points)
            field = torch.mul(field, label)
            multifield = multifield + field
        return multifield

