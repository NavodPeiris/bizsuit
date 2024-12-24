import random
from DemographicDataUtil import DemoGraphicDataUtil
from stream import UserActivity

class UserActivityUtil:
    activities = ["Click", "Purchase"]

    @staticmethod
    def generate_user_activity():
        return UserActivity(
            id=DemoGraphicDataUtil.get_user_id(),
            campaign_id=random.randint(0, 19),
            order_id=random.randint(0, 9999),
            amount=random.randint(0, 9999),
            units=random.randint(0, 99),
            activity=random.choice(UserActivityUtil.activities)
        )

    @staticmethod
    def tags():
        return {
            "activity": random.choice(UserActivityUtil.activities)
        }
