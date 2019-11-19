import numpy as np

class PanPy:
    def __init__(self):
        pass

    def deg_to_rad(self, degree):
        """
        convert degrees to radians
        """
        return degree * (np.pi / 180)

    def rad_to_deg(self, rad):
        """
        convert radians to degrees
        """
        return round(rad * (180 / np.pi))

    def pan_trig(self, offset):
        """
        Using the Trigonometric Panpot Law, returns a tuple pair of left and right 
        gain.

        : type offset : int
        : param offset : ( -100 <= value <= 100 ) -100 being all the way left and 100 being all the way right
        """
        degOffset = offset * 0.3
        theta0 = np.pi/2
        theta = np.array([-theta0, self.deg_to_rad(degOffset), theta0])
        gL = np.sin(np.pi*np.abs(theta-theta0)/(4*theta0))
        gR = np.sin(np.pi*(theta+theta0)/(4*theta0))
        return gL[1] , gR[1]

    def pan_sqrt(self, offset):
        """
        Using the Square Root Panpot Law, returns a tuple pair of left and right 
        gain.

        : type offset : int
        : param offset : ( -100 <= value <= 100 ) -100 being all the way left and 100 being all the way right        
        """
        degOffset = offset * 0.3
        theta0 = np.pi/2
        theta = np.array([-theta0, self.deg_to_rad(degOffset), theta0])
        gL = np.sqrt(np.abs(theta-theta0)/(2*theta0))
        gR = np.sqrt((theta+theta0)/(2*theta0))
        return gL[1] , gR[1]

    def pan_lin(self, offset):
        """
        Using the Linear Panpot Law, returns a tuple pair of left and right 
        gain.

        : type offset : int
        : param offset : ( -100 <= value <= 100 ) -100 being all the way left and 100 being all the way right 
        """
        degOffset = offset * 0.3
        theta0 = np.pi/2
        theta = np.array([-theta0, self.deg_to_rad(degOffset), theta0])
        gL = np.abs(theta-theta0)/(2*theta0)
        gR = (theta+theta0)/(2*theta0)
        return gL[1] , gR[1]