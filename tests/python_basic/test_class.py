class MyClass(object):
    def __init__(self):
        self.x = 1
    
    def add_y(self):
        self.y = 2

cls = MyClass()

print(cls.x)

cls.add_y()

print(cls.x, cls.y)