import math

import cv2
import matplotlib
import numpy as np

eps = 0.01


def alpha_blend_color(color, alpha):
    """blend color according to point conf"""
    return [int(c * alpha) for c in color]


def smart_width(d):
    if d < 5:
        return 1
    elif d < 10:
        return 2
    elif d < 20:
        return 3
    elif d < 40:
        return 4
    elif d < 80:
        return 5
    elif d < 160:
        return 6
    elif d < 320:
        return 7
    else:
        return 8


def draw_bodypose(
    canvas, candidate, subset, score, dynamic_width=False, dynamic_color=True
):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    limbSeq = [
        [2, 3],
        [2, 6],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [2, 9],
        [9, 10],
        [10, 11],
        [2, 12],
        [12, 13],
        [13, 14],
        [2, 1],
        [1, 15],
        [15, 17],
        [1, 16],
        [16, 18],
        [3, 17],
        [6, 18],
    ]

    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            conf = score[n][np.array(limbSeq[i]) - 1]
            if conf[0] < 0.3 or conf[1] < 0.3:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))

            if dynamic_width:
                width = smart_width(length) * 2
            else:
                width = stickwidth

            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), width), int(angle), 0, 360, 1
            )

            if dynamic_color:
                color = alpha_blend_color(colors[i], conf[0] * conf[1])
            else:
                color = colors[i]

            cv2.fillConvexPoly(canvas, polygon, color)

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            conf = score[n][i]
            x = int(x * W)
            y = int(y * H)
            if dynamic_color:
                color = alpha_blend_color(colors[i], conf)
            else:
                color = colors[i]

            cv2.circle(canvas, (x, y), 4, color, thickness=-1)

    return canvas


def draw_handpose(
    canvas, all_hand_peaks, all_hand_scores, dynamic_width=False, dynamic_color=True
):
    H, W, C = canvas.shape

    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ]

    for peaks, scores in zip(all_hand_peaks, all_hand_scores):

        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            score = int(scores[e[0]] * scores[e[1]] * 255)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:

                if dynamic_width:
                    length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                    width = smart_width(length) * 2
                else:
                    width = 2

                if dynamic_color:
                    color_weight = score
                else:
                    color_weight = 255

                cv2.line(
                    canvas,
                    (x1, y1),
                    (x2, y2),
                    matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])
                    * color_weight,
                    thickness=width,
                )

        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            x = int(x * W)
            y = int(y * H)
            score = int(scores[i] * 255)
            if dynamic_color:
                color = (0, 0, score)
            else:
                color = (0, 0, 255)

            if dynamic_width:
                radius = 3
            else:
                radius = 4

            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), radius, color, thickness=-1)
    return canvas


def draw_facepose(canvas, all_lmks, all_scores):
    H, W, C = canvas.shape
    for lmks, scores in zip(all_lmks, all_scores):
        for lmk, score in zip(lmks, scores):
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            conf = int(score * 255)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 3, (conf, conf, conf), thickness=-1)
    return canvas


def draw_pose(pose, H, W, include_body, include_hand, include_face, ref_w=2160):
    """vis dwpose outputs

    Args:
        pose (List): DWposeDetector outputs in dwpose_detector.py
        H (int): height
        W (int): width
        ref_w (int, optional) Defaults to 2160.

    Returns:
        np.ndarray: image pixel value in RGB mode
    """
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]

    sz = min(H, W)
    sr = (ref_w / sz) if sz != ref_w else 1

    ########################################## create zero canvas ##################################################
    canvas = np.zeros(shape=(int(H * sr), int(W * sr), 3), dtype=np.uint8)

    ########################################### draw body pose #####################################################
    if include_body:
        canvas = draw_bodypose(canvas, candidate, subset, score=bodies["score"])

    ########################################### draw hand pose #####################################################
    if include_hand:
        canvas = draw_handpose(canvas, hands, pose["hands_score"])

    ########################################### draw face pose #####################################################
    if include_face:
        canvas = draw_facepose(canvas, faces, pose["faces_score"])

    return cv2.cvtColor(cv2.resize(canvas, (W, H)), cv2.COLOR_BGR2RGB).transpose(
        2, 0, 1
    )


def draw_pose_musepose(
    pose, H, W, include_body, include_hand, include_face, ref_w=2160
):
    """vis dwpose outputs for MusePose

    Args:
        pose (List): DWposeDetector outputs in dwpose_detector.py
        H (int): height
        W (int): width
        ref_w (int, optional) Defaults to 2160.

    Returns:
        np.ndarray: image pixel value in RGB mode
    """
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]

    # only the most significant person
    faces = pose["faces"][:1]
    hands = pose["hands"][:2]
    candidate = bodies["candidate"][:18]
    subset = bodies["subset"][:1]

    sz = min(H, W)
    sr = (ref_w / sz) if sz != ref_w else 1

    ########################################## create zero canvas ##################################################
    canvas = np.zeros(shape=(int(H * sr), int(W * sr), 3), dtype=np.uint8)

    ########################################### draw body pose #####################################################
    if include_body:
        canvas = draw_bodypose(
            canvas,
            candidate,
            subset,
            score=bodies["score"],
            dynamic_color=False,
            dynamic_width=True,
        )

    ########################################### draw hand pose #####################################################
    if include_hand:
        canvas = draw_handpose(
            canvas, hands, pose["hands_score"], dynamic_color=False, dynamic_width=True
        )

    ########################################### draw face pose #####################################################
    if include_face:
        canvas = draw_facepose(canvas, faces, pose["faces_score"])

    return cv2.cvtColor(cv2.resize(canvas, (W, H)), cv2.COLOR_BGR2RGB).transpose(
        2, 0, 1
    )
