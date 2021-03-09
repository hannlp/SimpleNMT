class Test:
    def __init__(self):
        self.a = 1
    
    def func1():
        pass

test = Test()
print(hasattr(test, 'a'))
print(hasattr(test, 'b'))
print(hasattr(test, 'func1'))