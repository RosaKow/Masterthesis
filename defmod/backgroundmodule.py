class BackgroundModule(DeformationModule):
    """Module generating sum of translations."""
    def __init__(self, dim, nb_pts, sigma):
        super().__init__()
        self.__sigma = sigma
        self.__dim = dim
        self.__nb_pts = nb_pts
        
        
        
    # do i need @property ???
    
    def __call__(self, gd, points) :
        """Applies the generated vector field on given points."""
        
        
        return 