class UserDemoGraphicData:
    def __init__(self, i=None, age=None, gender=None, state=None, country=None):
        self.i = i
        self.age = age
        self.gender = gender
        self.state = state
        self.country = country

    def get_country(self):
        return self.country

    def set_country(self, country):
        self.country = country

    def get_i(self):
        return self.i

    def set_i(self, i):
        self.i = i

    def get_age(self):
        return self.age

    def set_age(self, age):
        self.age = age

    def get_gender(self):
        return self.gender

    def set_gender(self, gender):
        self.gender = gender

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state
