import requests
import numpy as np
from LSBSteg import decode
from keras.models import load_model

api_base_url = "http://3.70.97.142:5000"
# api_base_url = "http://localhost:5000"
team_id = "18u44K7"

model = load_model("Solvers/best_model_7.keras")


def init_eagle(team_id):
    """
    In this fucntion you need to hit to the endpoint to start the game as an eagle with your team id.
    If a sucessful response is returned, you will recive back the first footprints.
    """
    data = {"teamId": team_id}

    res = requests.post(url=f"{api_base_url}/eagle/start", json=data)
    res = res.json()

    footPrints = res["footprint"]

    print("==============Game Started===================")

    return footPrints


def image_preprocessing(image):
    image[np.isinf(image)] = float(0)
    image = np.expand_dims(image, axis=0)
    return image


def select_channel(footprints):
    """
    According to the footprint you recieved (one footprint per channel)
    you need to decide if you want to listen to any of the 3 channels or just skip this message.
    Your goal is to try to catch all the real messages and skip the fake and the empty ones.
    Refer to the documentation of the Footprints to know more what the footprints represent to guide you in your approach.
    """
    # footPrint1 = np.array(footprints["1"])
    # footPrint2 = np.array(footprints["2"])
    # footPrint3 = np.array(footprints["3"])

    # footPrint1 = image_preprocessing(footPrint1)
    # footPrint2 = image_preprocessing(footPrint2)
    # footPrint3 = image_preprocessing(footPrint3)

    ls = [footprints["1"], footprints["2"], footprints["3"]]

    print("============Selecting Channel==========")

    for i, img in enumerate(ls):
        img = image_preprocessing(np.array(img))
        pred = (model.predict(img) >= 0.5).astype(int).flatten()[0]

        if pred == 1:
            print(i + 1)
            return int(i + 1)

    print("No valid channel selected")
    print("=======================================")
    return int(-1)

    # print("============Selecting Channel==========")
    # for img in ls:
    #     preds.append(model.predict(img))

    # best = preds.index(max(preds))

    # print("All Predictions")
    # print(preds)

    # if preds[best] >= 0.5:
    #     print("Predicted Channel Id")
    #     print(best + 1)
    #     return best + 1
    # else:
    #     print("=======================================")
    #     return -1


def skip_msg(team_id):
    """
    If you decide to NOT listen to ANY of the 3 channels then you need to hit the end point skipping the message.
    If sucessful request to the end point , you will expect to have back new footprints IF ANY.
    """
    data = {"teamId": team_id}
    res = requests.post(url=f"{api_base_url}/eagle/skip-message", json=data)

    if res.text[0] == "{":
        res = res.json()
        nextFootPrints = res["nextFootprint"]

        hasFinished = False
        return nextFootPrints, hasFinished
    else:
        res = res.text
        print("No more FootPrints")

        hasFinished = True
        return res, hasFinished


def request_msg(team_id, channel_id):
    """
    If you decide to listen to any of the 3 channels then you need to hit the end point of selecting a channel to hear on (1,2 or 3)
    """
    data = {"teamId": team_id, "channelId": channel_id}

    res = requests.post(url=f"{api_base_url}/eagle/request-message", json=data)
    res = res.json()

    encodedMessage = res["encodedMsg"]

    encodedMessage = np.array(encodedMessage)

    return encodedMessage


def submit_msg(team_id, decoded_msg):
    """
    In this function you are expected to:
        1. Decode the message you requested previously
        2. call the api end point to send your decoded message
    If sucessful request to the end point , you will expect to have back new footprints IF ANY.
    """
    data = {"teamId": team_id, "decodedMsg": decoded_msg}
    res = requests.post(url=f"{api_base_url}/eagle/submit-message", json=data)

    if res.text[0] == "{":
        res = res.json()
        nextFootPrints = res["nextFootprint"]

        hasFinished = False
        return nextFootPrints, hasFinished
    else:
        res = res.text
        print("No more FootPrints")
        hasFinished = True
        return res, hasFinished


def end_eagle(team_id):
    """
    Use this function to call the api end point of ending the eagle  game.
    Note that:
    1. Not calling this fucntion will cost you in the scoring function
    """
    data = {"teamId": team_id}

    res = requests.post(url=f"{api_base_url}/eagle/end-game", json=data)
    res = res.text

    print("============The Eagle Game Ended===============")
    print(res)
    print("===============================================")


def submit_eagle_attempt(team_id):
    """
    Call this function to start playing as an eagle.
    You should submit with your own team id that was sent to you in the email.
    Remeber you have up to 15 Submissions as an Eagle In phase1.
    In this function you should:
       1. Initialize the game as fox
       2. Solve the footprints to know which channel to listen on if any.
       3. Select a channel to hear on OR send skip request.
       4. Submit your answer in case you listened on any channel
       5. End the Game
    """
    # Start the game
    footPrints = init_eagle(team_id)
    hasFinished = False

    # loop until you finish all footprints
    while True:
        channelId = select_channel(footPrints)
        print("Predicted Channel")
        print(channelId)

        if channelId != -1:
            image_msg = request_msg(team_id, channelId)
            decoded_msg = decode(image_msg)
            print("Message")
            print(decoded_msg)

            footPrints, hasFinished = submit_msg(team_id, decoded_msg)
        else:
            print("Skipped Message")
            footPrints, hasFinished = skip_msg(team_id)

        if hasFinished:
            break

    # End the game
    end_eagle(team_id)


# submit_eagle_attempt(team_id)
