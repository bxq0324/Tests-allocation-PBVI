
class AlphaVector(object):
    """
    Simple wrapper for the alpha vector used for representing the value function for a POMDP as a piecewise-linear,
    convex function
    """
    def __init__(self, a, v, z):
        self.action = a
        self.v = v
        self.z= z

    def copy(self):
        return AlphaVector(self.action, self.v, self.z)