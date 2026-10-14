import cv2
import mediapipe as mp
import numpy as np
import random
import time
import math
import pygame

# 🔊 Initialize sound
pygame.mixer.init()
shoot_sound = pygame.mixer.Sound("shoot.wav")
pygame.mixer.music.load("bg.mp3")
pygame.mixer.music.play(-1)

# Load target image (face)
target_img = cv2.imread("target.png", cv2.IMREAD_UNCHANGED)
# 👉 Resize (width, height)
target_img = cv2.resize(target_img, (80, 80))

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

score = 0

# Target position & speed
target_x = 0
target_y = 200
target_speed = 5

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def overlay_image(bg, img, x, y):
    h, w, _ = img.shape

    if x + w > bg.shape[1]:
        x = bg.shape[1] - w
    if y + h > bg.shape[0]:
        y = bg.shape[0] - h

    for i in range(h):
        for j in range(w):
            if img[i, j][3] != 0:
                bg[y + i, x + j] = img[i, j][:3]

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    cursor_x, cursor_y = 0, 0
    shoot = False

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            if len(lm_list) > 8:
                cursor_x, cursor_y = lm_list[8]

                if distance(lm_list[4], lm_list[8]) < 40:
                    shoot = True

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # 🎯 Move target (left → right loop)
    target_x += target_speed
    if target_x > w:
        target_x = -100

    # Increase speed with score
    target_speed = 5 + score * 0.5

    # Draw target image
    overlay_image(frame, target_img, int(target_x), target_y)

    # Draw cursor
    cv2.circle(frame, (cursor_x, cursor_y), 10, (255, 0, 0), -1)

    # Shooting logic
    if shoot:
        shoot_sound.play()

        # Hit detection (approx box check)
        if (target_x < cursor_x < target_x + 100 and
            target_y < cursor_y < target_y + 100):
            score += 1
            time.sleep(0.2)

    # Score display
    cv2.putText(frame, f"Score: {score}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Shooting Game", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()