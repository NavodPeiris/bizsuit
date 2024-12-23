import random

class DemoGraphicDataUtil:
    @staticmethod
    def get_user_id():
        return random.randint(0, 100000)

    @staticmethod
    def get_country():
        countries = ["USA", "JAPAN", "CANADA", "INDIA", "SOUTH_AFRICA"]
        return random.choice(countries)

    @staticmethod
    def get_state(country):
        country_states = {
            "USA": ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", 
                    "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", 
                    "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri"],
            "INDIA": ["Andhra", "Amaravati", "Kurnool", "Arunachal", "Assam", "Bihar", "Chhattisgarh", "Goa", 
                      "Gujarat", "Haryana", "Himachal", "Dharamshala", "Jharkhand", "Karnataka", "Kerala", "Madhya", 
                      "Maharashtra", "Nagpur", "Manipur"],
            "CANADA": ["Ontario", "Nova_Scotia", "Manitoba", "Prince_Edward", "Newfoundland_Labrador"],
            "SOUTH_AFRICA": ["Eastern_Cape", "Free_State", "Gauteng", "KwaZulu-Natal", "Limpopo", "Mpumalanga", 
                             "Northern_Cape", "North_West"],
            "JAPAN": ["Hokkaido", "Tohoku", "Kanto", "Chubu", "Kinki_Kansai", "Chugoku", "Shikoku", "Kyushu"]
        }
        states = country_states.get(country, [])
        if not states:
            raise ValueError(f"No states found for country: {country}")
        return random.choice(states)

    @staticmethod
    def get_age():
        return random.randint(0, 60)

    @staticmethod
    def get_gender():
        genders = ["MALE", "FEMALE"]
        return random.choice(genders)
