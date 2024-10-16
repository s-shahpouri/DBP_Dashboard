import copy

class Coordinate(object): 
    """"""
    __slots__ = ('x', 'y', 'z', 'roll', 'pitch', 'yaw', 'rotation_center') #rotation_center is [x,y,x]

    axes = ('x', 'y', 'z', 'roll', 'pitch', 'yaw')

    def __init__(self, x, y, z, roll, pitch, yaw, rotation_center = {'x': 0, 'y': 0, 'z': 0}):
        """"""
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.rotation_center = rotation_center

    def __getitem__(self, key):
        return self.__getattribute__(key)
    
    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __repr__(self):
        """"""
        return 'Coordinate({}, {}, {}, {}, {}, {}, {})'.format(self.x, self.y, self.z, self.roll, self.pitch, self.yaw, self.rotation_center)

    
    def rigid_trafo_dict(self):
        return {
            'YawDegrees': self.yaw,
            'PitchDegrees': self.pitch,
            'RollDegrees': self.roll,
            'Translation': {'x': self.x, 'y': self.y, 'z': self.z},
            'RotationCenter': {'x': self.rotation_center['x'], 'y': self.rotation_center['y'], 'z': self.rotation_center['z']}
        }
    
    def same_rotation(self, coord):
        return self.roll == coord.roll and self.pitch == coord.pitch and self.yaw == coord.yaw and self.rotation_center == coord.rotation_center
    
    def __add__(self, other):
        #assuming rotation center is the same for both. With different rotation centers still has to be implemented
        assert self.rotation_center == other.rotation_center, "cannot add coordinates because they don't have the same rotation center"
        result = copy.deepcopy(self)
        for axis in self.axes:
            result[axis] += other[axis]
        return result
    
    # def __mul__(self, other):
    #     #assuming rotation center is the same for both. With different rotation centers still has to be implemented
    #     if isinstance(other, int) or isinstance(other, float):
    #         result = copy.deepcopy(self)
    #         for axis in self.axes:
    #             result[axis] *= other
    #         return result
        
    # def __rmul__(self, other):
    #         result = self * other
    #         return result
    
    def __sub__(self, other):
        #assuming rotation center is the same for both. With different rotation centers still has to be implemented
        assert self.rotation_center == other.rotation_center, "cannot add coordinates because they don't have the same rotation center"
        result = copy.deepcopy(self)
        for axis in self.axes:
            result[axis] -= other[axis]
        return result
    
    def __neg__(self):
        result = copy.deepcopy(self)
        for axis in self.axes:
            result[axis] = -result[axis]
        return result