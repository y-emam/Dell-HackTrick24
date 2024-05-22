import requests
import numpy as np
from LSBSteg import encode
from riddle_solvers import riddle_solvers

api_base_url = "http://3.70.97.142:5000"
# api_base_url = "http://localhost:5000"
# prof_base_url = "http://16..171.171.147:5000"
team_id = "18u44K7"


def init_fox(team_id):
    """
    In this fucntion you need to hit to the endpoint to start the game as a fox with your team id.
    If a sucessful response is returned, you will recive back the message that you can break into chunkcs
      and the carrier image that you will encode the chunk in it.
    """
    data = {"teamId": team_id}

    res = requests.post(url=f"{api_base_url}/fox/start", json=data)
    res = res.json()

    message = res["msg"]
    image_carrier = res["carrier_image"]

    img_np = np.array(image_carrier)
    print("==============Game Started===================")
    print(img_np.shape)
    print(message)
    print("=============================================")

    return image_carrier, message


def split_message(message):
    """
    In this function I will split the message into chunks
    """
    message1 = message[0:7]
    message2 = message[7:14]
    message3 = message[14:]

    return [message1, message2, message3]


def generate_message_array(message, image_carrier):
    """
    In this function you will need to create your own startegy. That includes:
        1. How you are going to split the real message into chunkcs
        2. Include any fake chunks
        3. Decide what 3 chuncks you will send in each turn in the 3 channels & what is their entities (F,R,E)
        4. Encode each chunck in the image carrier
    """
    # FFR - FFR - FFR
    splited_message = split_message(
        message
    )  # in this case an array of 3 messages ex: 'Secret' ' messages' ' rock'
    n_chunks = 3
    entities = [["F", "F", "R"], ["F", "F", "R"], ["F", "F", "R"]]
    image_carrier = np.array(image_carrier)

    for i in range(n_chunks):
        img1 = image_carrier.copy()
        real = encode(img1, splited_message[i])
        img2 = image_carrier.copy()
        fake1 = encode(img2, "FAKE")
        img3 = image_carrier.copy()
        fake2 = encode(img3, "FAKE")

        send_message(
            team_id=team_id,
            messages=[fake1.tolist(), fake2.tolist(), real.tolist()],
            message_entities=entities[i],
        )


def get_riddle(team_id, riddle_id):
    """
    In this function you will hit the api end point that requests the type of riddle you want to solve.
    use the riddle id to request the specific riddle.
    Note that:
        1. Once you requested a riddle you cannot request it again per game.
        2. Each riddle has a timeout if you didnot reply with your answer it will be considered as a wrong answer.
        3. You cannot request several riddles at a time, so requesting a new riddle without answering the old one
          will allow you to answer only the new riddle and you will have no access again to the old riddle.
    """
    data = {"teamId": team_id, "riddleId": riddle_id}
    res = requests.post(url=f"{api_base_url}/fox/get-riddle", json=data)
    res = res.json()

    test_case = res["test_case"]

    return test_case


def solve_riddle(team_id, riddleId, test_case):
    """
    In this function you will solve the riddle that you have requested.
    You will hit the API end point that submits your answer.
    Use te riddle_solvers.py to implement the logic of each riddle.
    """

    solver = riddle_solvers[riddleId]
    solution = solver(test_case)

    data = {"teamId": team_id, "solution": solution}

    res = requests.post(url=f"{api_base_url}/fox/solve-riddle", json=data)
    res = res.json()

    print("================================")
    print(f"Riddle: {riddleId}")
    print(f"Status: {res['status']}")
    print(f"Budget Increase: {res['budget_increase']}")
    print(f"Total Budget: {res['total_budget']}")
    print("================================")


def send_message(team_id, messages, message_entities):
    """
    Use this function to call the api end point to send one chunk of the message.
    You will need to send the message (images) in each of the 3 channels along with their entites.
    Refer to the API documentation to know more about what needs to be send in this api call.
    """
    print("TeamId: ")
    print(team_id)
    print("Message Entities: ")
    print(message_entities)

    data = {
        "teamId": team_id,
        "messages": messages,
        "message_entities": message_entities,
    }
    res = requests.post(url=f"{api_base_url}/fox/send-message", json=data)
    res = res.json()

    status = res["status"]

    print(f"Message sent? : {status}")

    return status


def end_fox(team_id):
    """
    Use this function to call the api end point of ending the fox game.
    Note that:
    1. Not calling this fucntion will cost you in the scoring function
    2. Calling it without sending all the real messages will also affect your scoring fucntion
      (Like failing to submit the entire message within the timelimit of the game).
    """
    data = {"teamId": team_id}

    res = requests.post(url=f"{api_base_url}/fox/end-game", json=data)
    res = res.text

    print("============The Game Ended===============")
    print(res)
    print("=========================================")


def submit_fox_attempt(team_id):
    """
     Call this function to start playing as a fox.
     You should submit with your own team id that was sent to you in the email.
     Remeber you have up to 15 Submissions as a Fox In phase1.
     In this function you should:
        1. Initialize the game as fox
        2. Solve riddles
        3. Make your own Strategy of sending the messages in the 3 channels
        4. Make your own Strategy of splitting the message into chunks
        5. Send the messages
        6. End the Game
    Note that:
        1. You HAVE to start and end the game on your own. The time between the starting and ending the game is taken into the scoring function
        2. You can send in the 3 channels any combination of F(Fake),R(Real),E(Empty) under the conditions that
            2.a. At most one real message is sent
            2.b. You cannot send 3 E(Empty) messages, there should be atleast R(Real)/F(Fake)
        3. Refer To the documentation to know more about the API handling
    """
    # Start the game
    image_carrier, message = init_fox(team_id)

    # Solve the Riddles
    allRiddles = [
        "problem_solving_easy",
        "problem_solving_medium",
        "problem_solving_hard",
        "sec_hard",
        "ml_medium",
    ]

    for riddleId in allRiddles:
        test_case = get_riddle(team_id=team_id, riddle_id=riddleId)
        solve_riddle(team_id, riddleId=riddleId, test_case=test_case)

    # Send the messages
    generate_message_array(message, image_carrier)

    # End the Game
    end_fox(team_id)
    pass


# submit_fox_attempt(team_id)
