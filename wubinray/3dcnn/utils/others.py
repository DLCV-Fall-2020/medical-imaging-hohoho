
class Averager():
    def __init__(self):
        self.clean()

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v
    
    def clean(self):
        self.n = 0
        self.v = 0

