class UserActivity:
    def __init__(self, id=None, campaign_id=None, order_id=None, amount=None, units=None, activity=None):
        self.id = id
        self.campaign_id = campaign_id
        self.order_id = order_id
        self.amount = amount
        self.units = units
        self.activity = activity

    def get_id(self):
        return self.id

    def set_id(self, id):
        self.id = id

    def get_campaign_id(self):
        return self.campaign_id

    def set_campaign_id(self, campaign_id):
        self.campaign_id = campaign_id

    def get_order_id(self):
        return self.order_id

    def set_order_id(self, order_id):
        self.order_id = order_id

    def get_amount(self):
        return self.amount

    def set_amount(self, amount):
        self.amount = amount

    def get_units(self):
        return self.units

    def set_units(self, units):
        self.units = units

    def get_activity(self):
        return self.activity

    def set_activity(self, activity):
        self.activity = activity
