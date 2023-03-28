from datetime import datetime
import gspread

def save_user_data(user_data):
    """
    Save user data to a google sheet using gspread
    :param user_data: dictionary containing user data
    """
    # connect to google sheets using gspread with credentials
    account = gspread.service_account(filename='data_utils/calibrated-predictions-survey-4bb73ccb8ec5.json')
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
