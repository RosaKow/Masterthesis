import math
pi = math.pi
import torch

class RegistrationData:
    """ Builds different kind of datasets for matching"""
    def __init__(self):
        super().__init__()

        
class PointCircles(RegistrationData):
    """ Builds source and target points for translation and scaling of circles """
    def __init__(self):
        super().__init__()        
        self.__source = None
        self.__target = None
        
    @property
    def source(self):
        return self.__source
    
    @property
    def target(self):
        return self.__target
    
    
    def CirclePoints(self, origin, r,n):
        points = []
        for x in range (0,n):
            points.append([origin[0] + math.cos(2*pi/n*x)*r,origin[1] + math.sin(2*pi/n*x)*r])
        return torch.tensor(points, requires_grad=True)

    def multipleCircles(self, origin, radius, numberPoints):
        circles = []
        for o,r,n in zip(origin, radius, numberPoints):
            circles.append(self.CirclePoints(o, r, n))
        return circles
    
    def __call__(self, nb_pts, origin1, radius1, transvec, scal=1):
        origin1 = torch.tensor(origin1)
        origin2 = [o + torch.tensor(v) for o,v in zip(origin1, transvec)]
        radius2 = radius1 * scal
        self.__source = self.multipleCircles(origin1, radius1, nb_pts)
        self.__target = self.multipleCircles(origin2, radius2, nb_pts)
