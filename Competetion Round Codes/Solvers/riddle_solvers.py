# Add the necessary imports here
import pandas as pd
import cv2
import numpy as np
import math
from transformers import pipeline
import torch

# from PIL import Image
# import torchvision.transforms as transforms
from utils import *

# import cv2
import matplotlib.pyplot as plt

from collections import Counter
import heapq
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn import preprocessing

data = pd.read_csv(
    "/content/HackTrick24/Riddles/ml_medium_dataset/MlMediumTrainingData.csv"
)
x = data[["x_", "y_"]]
y = data["class"].tolist()

class_count = data["class"].value_counts()
to_be_generated = class_count[0] - class_count[-1]
# slice from the original data the part that you want to test
org_x = x[:]
org_y = y[:]

# Generate a random classification dataset
gen_x1 = x["x_"].to_numpy()
gen_x2 = x["y_"].to_numpy()

x_zipped = zip(x["x_"].tolist(), x["y_"].tolist())
x_set = set(x_zipped)
gen = 0
while gen < to_be_generated:
    new_x = random.uniform(
        gen_x1.min() - 5 * gen_x1.mean(), gen_x1.max() + 5 * gen_x1.mean()
    )
    new_y = random.uniform(
        gen_x2.min() - 5 * gen_x2.mean(), gen_x2.max() + 5 * gen_x2.mean()
    )
    new_sample = (new_x, new_y)
    if not x_set.__contains__(new_sample):
        gen_x1 = np.append(gen_x1, new_sample[0])
        gen_x2 = np.append(gen_x2, new_sample[1])
        y.append(-1)
        gen += 1

x = pd.DataFrame({"x_": gen_x1, "y_": gen_x2})
y = np.array(y)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

encoder = preprocessing.LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)
org_y = encoder.fit_transform(org_y)

# Create an RBF SVM classifier with gamma=0.1
clf = SVC(kernel="rbf", gamma=0.1)

# Train the classifier on the training data
clf.fit(X_train, y_train)


def calculate_similarity(shred1, shred2):
    """
    Calculate similarity between the rightmost pixel of shred1 and the leftmost pixel of shred2.
    """
    return np.sum(shred1[:, -1] == shred2[:, 0])


def solve_cv_easy(test_case: tuple) -> list:

    shredded_image, shred_width = test_case
    shredded_image = np.array(shredded_image)
    """
    This function takes a tuple as input and returns a list as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A numpy array representing a shredded image.
        - An integer representing the shred width in pixels.

    Returns:
    list: A list of integers representing the order of shreds. When combined in this order, it builds the whole image.
    """
    num_shreds = shredded_image.shape[1] // shred_width

    similarity_matrix = np.zeros((num_shreds, num_shreds))
    for i in range(num_shreds):
        for j in range(num_shreds):
            if i != j:
                similarity_matrix[i, j] = calculate_similarity(
                    shredded_image[:, i * shred_width : (i + 1) * shred_width],
                    shredded_image[:, j * shred_width : (j + 1) * shred_width],
                )

    used = [False] * num_shreds

    ordered_indices = [0]
    used[0] = True

    while len(ordered_indices) < num_shreds:
        last_shred = ordered_indices[-1]
        max_similarity = -1
        next_shred = -1
        for i in range(num_shreds):
            if not used[i] and similarity_matrix[last_shred, i] > max_similarity:
                max_similarity = similarity_matrix[last_shred, i]
                next_shred = i
        ordered_indices.append(next_shred)
        used[next_shred] = True

    return ordered_indices


def solve_cv_medium(input: tuple) -> list:
    """
    This function takes a tuple as input and returns a list as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A numpy array representing the RGB base image.
        - A numpy array representing the RGB patch image.

    Returns:
    list: A list representing the real image.
    """
    combined_image_array, patch_image_array = input
    combined_image = np.array(combined_image_array, dtype=np.uint8)
    org_img = combined_image
    patch_image = np.array(patch_image_array, dtype=np.uint8)

    # Convert images to grayscale
    combined_gray = cv2.cvtColor(combined_image, cv2.COLOR_RGB2GRAY)
    patch_gray = cv2.cvtColor(patch_image, cv2.COLOR_RGB2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(combined_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(patch_gray, None)

    # Initialize FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Find matches between descriptors
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )

        # Find homography matrix
        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        # Get coordinates of the patch corners
        h, w = patch_gray.shape
        patch_corners = np.array(
            [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]], dtype=np.float32
        )
        patch_corners = np.array([patch_corners])

        # Warp patch corners to align with base image
        warped_corners = cv2.perspectiveTransform(patch_corners, H)

        # Extract top-left and bottom-right coordinates of the warped patch
        tl = np.min(warped_corners, axis=1)[0]
        br = np.max(warped_corners, axis=1)[0]

        # Remove the patch from the combined image
        combined_image[int(tl[1]) : int(br[1]), int(tl[0]) : int(br[0])] = 0

        # Interpolate the missing area
        mask = np.zeros_like(combined_gray)
        mask[int(tl[1]) : int(br[1]), int(tl[0]) : int(br[0])] = 255
        combined_image = cv2.inpaint(
            combined_image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA
        )

    return combined_image.tolist()


def solve_cv_hard(input: tuple) -> int:
    extracted_question, image = input
    """
    This function takes a tuple as input and returns an integer as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A string representing a question about an image.
        - An RGB image object loaded using the Pillow library.

    Returns:
    int: An integer representing the answer to the question about the image.
    """
    vqa_pipeline = pipeline("visual-question-answering")

    return int(vqa_pipeline(image, extracted_question, top_k=1)[0]["answer"])


def solve_ml_easy(input: pd.DataFrame) -> list:
    data = pd.DataFrame(data)

    """
    This function takes a pandas DataFrame as input and returns a list as output.

    Parameters:
    input (pd.DataFrame): A pandas DataFrame representing the input data.

    Returns:
    list: A list of floats representing the output of the function.
    """
    return []


def solve_ml_medium(input: list) -> int:
    """
    This function takes a list as input and returns an integer as output.

    Parameters:
    input (list): A list of signed floats representing the input data.

    Returns:
    int: An integer representing the output of the function.
    """

    lst = input[0]
    clf = input[1]
    xpred = pd.DataFrame({"x_": [lst[0]], "y_": [lst[1]]})
    y_pred = clf.predict(xpred)
    y_pred = 0 if y_pred[0] == 1 else -1

    print("label : ", y_pred)
    return y_pred


def solve_sec_medium(input) -> str:
    """
    This function takes a torch.Tensor as input and returns a string as output.

    Parameters:
    input (torch.Tensor): A torch.Tensor representing the image that has the encoded message.

    Returns:
    str: A string representing the decoded message from the image.
    """
    try:
        # data preprocessing
        input = np.array(input, dtype=np.float32)
        input = np.transpose(input, (2, 0, 1))
        input = torch.tensor(input)

        # logic
        batched_image_tensor = torch.stack(list(input))
        batched_image_tensor = batched_image_tensor.unsqueeze(0)
        decoded_message = decode(batched_image_tensor)
        return decoded_message
    except Exception as e:
        print(str(e))
        return ""


def solve_sec_hard(input: tuple) -> str:
    pc1 = [
        56,
        48,
        40,
        32,
        24,
        16,
        8,
        0,
        57,
        49,
        41,
        33,
        25,
        17,
        9,
        1,
        58,
        50,
        42,
        34,
        26,
        18,
        10,
        2,
        59,
        51,
        43,
        35,
        62,
        54,
        46,
        38,
        30,
        22,
        14,
        6,
        61,
        53,
        45,
        37,
        29,
        21,
        13,
        5,
        60,
        52,
        44,
        36,
        28,
        20,
        12,
        4,
        27,
        19,
        11,
        3,
    ]
    pc2 = [
        13,
        16,
        10,
        23,
        0,
        4,
        2,
        27,
        14,
        5,
        20,
        9,
        22,
        18,
        11,
        3,
        25,
        7,
        15,
        6,
        26,
        19,
        12,
        1,
        40,
        51,
        30,
        36,
        46,
        54,
        29,
        39,
        50,
        44,
        32,
        47,
        43,
        48,
        38,
        55,
        33,
        52,
        45,
        41,
        49,
        35,
        28,
        31,
    ]
    ip = [
        57,
        49,
        41,
        33,
        25,
        17,
        9,
        1,
        59,
        51,
        43,
        35,
        27,
        19,
        11,
        3,
        61,
        53,
        45,
        37,
        29,
        21,
        13,
        5,
        63,
        55,
        47,
        39,
        31,
        23,
        15,
        7,
        56,
        48,
        40,
        32,
        24,
        16,
        8,
        0,
        58,
        50,
        42,
        34,
        26,
        18,
        10,
        2,
        60,
        52,
        44,
        36,
        28,
        20,
        12,
        4,
        62,
        54,
        46,
        38,
        30,
        22,
        14,
        6,
    ]
    expansion_table = [
        31,
        0,
        1,
        2,
        3,
        4,
        3,
        4,
        5,
        6,
        7,
        8,
        7,
        8,
        9,
        10,
        11,
        12,
        11,
        12,
        13,
        14,
        15,
        16,
        15,
        16,
        17,
        18,
        19,
        20,
        19,
        20,
        21,
        22,
        23,
        24,
        23,
        24,
        25,
        26,
        27,
        28,
        27,
        28,
        29,
        30,
        31,
        0,
    ]
    S1 = [
        [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
        [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
        [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
        [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13],
    ]
    S2 = [
        [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
        [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
        [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
        [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9],
    ]
    S3 = [
        [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
        [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
        [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
        [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12],
    ]
    S4 = [
        [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
        [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
        [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
        [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14],
    ]
    S5 = [
        [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
        [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
        [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
        [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3],
    ]
    S6 = [
        [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
        [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
        [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
        [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13],
    ]
    S7 = [
        [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
        [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
        [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
        [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12],
    ]
    S8 = [
        [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
        [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
        [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
        [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11],
    ]
    p_sbox = [
        15,
        6,
        19,
        20,
        28,
        11,
        27,
        16,
        0,
        14,
        22,
        25,
        4,
        17,
        30,
        9,
        1,
        7,
        23,
        13,
        31,
        26,
        2,
        8,
        18,
        12,
        29,
        5,
        21,
        10,
        3,
        24,
    ]
    fp = [
        39,
        7,
        47,
        15,
        55,
        23,
        63,
        31,
        38,
        6,
        46,
        14,
        54,
        22,
        62,
        30,
        37,
        5,
        45,
        13,
        53,
        21,
        61,
        29,
        36,
        4,
        44,
        12,
        52,
        20,
        60,
        28,
        35,
        3,
        43,
        11,
        51,
        19,
        59,
        27,
        34,
        2,
        42,
        10,
        50,
        18,
        58,
        26,
        33,
        1,
        41,
        9,
        49,
        17,
        57,
        25,
        32,
        0,
        40,
        8,
        48,
        16,
        56,
        24,
    ]
    sboxs = [S1, S2, S3, S4, S5, S6, S7, S8]

    final_plain_txt = ""
    res = ""
    res2 = ""

    def toBinary(a):
        return bin(int(a, 16))[2:].zfill(64)

    def sbox(stable, i, ans):
        return stable[int(ans[i][0] + ans[i][5], 2)][int(ans[i][1:5], 2)]

    def permutation(size, old, pt):
        z = ""
        for i in range(0, size):
            z += old[pt[i]]
        return z

    plaintext = toBinary(input[1])

    def enc(final_plain_txt, kv):
        c = 1
        for i in range(1, 17):

            k = kv

            lpl = final_plain_txt[: 64 // 2]
            rpl = final_plain_txt[64 // 2 :]
            expan = permutation(48, rpl, expansion_table)
            ans = ""
            ans2 = ""
            ans3 = ""
            for j in range(0, 48):
                ans += str(int(k[i - 1][j]) ^ int(expan[j]))

            # ans = textwrap.wrap(ans, 6)
            ans = [ans[j : j + 6] for j in range(0, len(ans), 6)]
            for i in range(8):
                ans2 += format(sbox(sboxs[i], i, ans), "04b")
            ans2 = permutation(32, ans2, p_sbox)
            for j in range(0, 32):
                ans3 += str(int(lpl[j]) ^ int(ans2[j]))
            lpl = rpl
            rpl = ans3
            final_plain_txt = lpl + rpl
            c += 1

        return permutation(64, rpl + lpl, fp)

    def ksh():
        z = permutation(56, key, pc1)
        finalk = []
        for i in range(1, 17):

            l = z[: 56 // 2]
            r = z[56 // 2 :]

            if i == 1 or i == 2 or i == 9 or i == 16:
                l = l[1:] + l[0:1]
                r = r[1:] + r[0:1]

            else:
                l = l[2:] + l[0:2]
                r = r[2:] + r[0:2]

            z = l + r
            finalk.append(permutation(48, z, pc2))

        return finalk

    while True:
        key = input[0]

        if len(key) == 16:
            break
    key = toBinary(key)

    if len(plaintext) > 64:
        last = ""
        final_plain_txt = [plaintext[i : i + 64] for i in range(0, len(plaintext), 64)]
        zzz = len(final_plain_txt) - 1
        while len(final_plain_txt[zzz]) < 64:
            final_plain_txt[zzz] = f'{"0"}{final_plain_txt[zzz]}'

        for i in range(0, zzz + 1):
            n = ""
            last += final_plain_txt[i]
            final_plain_txt[i] = permutation(64, final_plain_txt[i], ip)
            n = enc(final_plain_txt[i], ksh())
            res += n
    else:
        while len(plaintext) < 64:
            plaintext = f'{"0"}{plaintext}'
        final_plain_txt = plaintext
        final_plain_txt = permutation(64, final_plain_txt, ip)
        res = enc(final_plain_txt, ksh())

    return hex(int(res, 2))[2:].upper()


def solve_problem_solving_easy(input) -> list:
    """
    This function takes a tuple as input and returns a list as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A list of strings representing a question.
        - An integer representing a key.

    Returns:
    list: A list of strings representing the solution to the problem.
    """
    strings, x = input

    # Count the occurrences of each string
    frq = Counter(strings)

    # Use a min heap to maintain the top x elements based on frequency and lexicographical order
    min_heap = [(-freq, word) for word, freq in frq.items()]
    heapq.heapify(min_heap)

    ans = []
    while len(ans) < x and min_heap:
        freq, word = heapq.heappop(min_heap)
        ans.append(word)

    return ans


def solve_problem_solving_medium(input: str) -> str:
    """
    This function takes a string as input and returns a string as output.

    Parameters:
    input (str): A string representing the input data.

    Returns:
    str: A string representing the solution to the problem.
    """
    s = input
    counts = []
    result = []
    current = ""
    count = 0

    for i in range(len(s)):
        if "0" <= s[i] <= "9":
            count = count * 10 + int(s[i])
        elif "a" <= s[i] <= "z":
            current += s[i]
        elif s[i] == "[":
            counts.append(count)
            result.append(current)
            current = ""
            count = 0
        elif s[i] == "]":
            repeat = counts.pop()
            prev = result.pop()

            for _ in range(repeat):
                prev += current
            current = prev

    return current


def solve_problem_solving_hard(input: tuple) -> int:
    """
    This function takes a tuple as input and returns an integer as output.

    Parameters:
    input (tuple): A tuple containing two integers representing m and n.

    Returns:
    int: An integer representing the solution to the problem.
    """
    n, m = input

    dp = [[0] * (m + 5) for _ in range(n + 5)]
    dp[n - 1][m - 1] = 1

    for j in range(m - 2, -1, -1):
        dp[n - 1][j] = dp[n - 1][j + 1]

    for i in range(n - 2, -1, -1):
        dp[i][m - 1] = dp[i + 1][m - 1]

    for i in range(n - 2, -1, -1):
        for j in range(m - 2, -1, -1):
            dp[i][j] = dp[i + 1][j] + dp[i][j + 1]

    return dp[0][0]


riddle_solvers = {
    "cv_easy": solve_cv_easy,
    "cv_medium": solve_cv_medium,
    "cv_hard": solve_cv_hard,
    "ml_easy": solve_ml_easy,
    "ml_medium": solve_ml_medium,
    "sec_medium_stegano": solve_sec_medium,
    "sec_hard": solve_sec_hard,
    "problem_solving_easy": solve_problem_solving_easy,
    "problem_solving_medium": solve_problem_solving_medium,
    "problem_solving_hard": solve_problem_solving_hard,
}
