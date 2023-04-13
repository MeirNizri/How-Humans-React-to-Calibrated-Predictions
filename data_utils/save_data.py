import gspread
import pymongo
import json
from datetime import datetime


def save_user_data_gspread(user_data):
    """
    Save user data to a google sheet using gspread
    :param user_data: dictionary containing user data
    """
    # connect to google sheets using gspread with credentials
    account = gspread.service_account(filename='data_utils/credentials.json')
    spreadsheet = account.open("Calibrated Predictions Survey")

    # get sheets from spreadsheet
    user_sheet = spreadsheet.worksheet("users")
    pred_sheet = spreadsheet.worksheet("predictions")
    labels_sheet = spreadsheet.worksheet("labels")
    pred_test_sheet = spreadsheet.worksheet("pred_test")
    labels_test_sheet = spreadsheet.worksheet("outcome_test")
    first_rate_sheet = spreadsheet.worksheet("first_rate")
    last_rate_sheet = spreadsheet.worksheet("last_rate")

    # set user id
    user_id = len(user_sheet.get_all_values())

    # add user_id to the first cell of each sheet row and the lists to the rest of the row
    user_sheet.append_row([user_id] + 
                          [user_data["model_name"]] +
                          [2023-int(user_data["year"])] + 
                          [user_data["gender"]] + 
                          [user_data["occupation"]] +
                          [user_data["review"]] +
                          [str(datetime.now())])
    pred_sheet.append_row([user_id] + user_data["predictions"])
    labels_sheet.append_row([user_id] + user_data["labels"])
    pred_test_sheet.append_row([user_id] + user_data["predTest"])
    labels_test_sheet.append_row([user_id] + user_data["outcomeTest"])
    first_rate_sheet.append_row([user_id] + user_data["firstRate"])
    last_rate_sheet.append_row([user_id] + user_data["lastRate"])
    

# save user data to a mongo database 
def save_user_data_mongo(user_data):
    """
    Save user data to a mongo database
    :param user_data: dictionary containing user data
    """
    # connect to mongo database
    client = pymongo.MongoClient("mongodb+srv://root:a1234@cluster0.wmqvoo6.mongodb.net/test")
    db = client["calibration_survey"]
    users = db["users"]

    # set user id
    user_id = users.count_documents({}) + 1

    # add user_id to the first cell of each sheet row and the lists to the rest of the row
    user_data["user_id"] = user_id
    user_data["date"] = str(datetime.now())
    users.insert_one(user_data)



# if run as main, read all documents from the users collection and save each document to a seperate text file with indentations
if __name__ == "__main__":
    # connect to mongo database
    client = pymongo.MongoClient("mongodb+srv://root:a1234@cluster0.wmqvoo6.mongodb.net/test")
    db = client["calibration_survey"]
    users = db["users"]

    # get all documents from the users collection
    user_data = users.find()

    # save each json object to a seperate text file with indentations
    for user in user_data:
        del user["_id"]
        with open(f"data_utils/user_data/{user['user_id']}.txt", "w") as f:
            json.dump(user, f, indent=4)

