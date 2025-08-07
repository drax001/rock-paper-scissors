from flask import Flask, render_template, Response, request, redirect, url_for
from ultralytics import YOLO
import cv2
import numpy as np
import random
import pygame
import os
from datetime import datetime

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8-rps.pt")
class_names = ["Paper", "Rock", "Scissors"]

# Game state
game_mode = "computer"
best_of = 3
player1_score = 0
player2_score = 0
final_winner = ""
round_done = False

# Initialize sound
pygame.mixer.init()

def play_sound(action):
    """Plays sound effects for moves and wins."""
    sounds = {
        "move": "sounds/move.wav",
        "win": "sounds/win.wav"
    }
    pygame.mixer.Sound(sounds[action]).play()

def get_winner(p1, p2):
    """Determines the winner based on Rock-Paper-Scissors rules."""
    if p1 == p2:
        return "Draw"
    elif (p1 == "Rock" and p2 == "Scissors") or \
         (p1 == "Paper" and p2 == "Rock") or \
         (p1 == "Scissors" and p2 == "Paper"):
        return "Player 1"
    else:
        return "Player 2"

def save_winning_frame(frame):
    """Saves an image of the winning round."""
    os.makedirs("static/saved_rounds", exist_ok=True)
    filename = f"static/saved_rounds/win_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(filename, frame)
    print(f"✅ Winning frame saved as {filename}")

@app.route("/", methods=["GET", "POST"])
def index():
    """Renders game setup page."""
    global game_mode, best_of, player1_score, player2_score, final_winner
    if request.method == "POST":
        game_mode = request.form.get("mode")
        best_of = int(request.form.get("rounds"))
        player1_score = 0
        player2_score = 0
        final_winner = ""
        return redirect(url_for("game"))
    return render_template("index.html")

@app.route("/game")
def game():
    """Renders the game page."""
    images = os.listdir("static/saved_rounds")[-4:]  # Show last 4 winning images
    return render_template("game.html", best_of=best_of, images=images, final_winner=final_winner)

@app.route("/video_feed")
def video_feed():
    """Handles live video feed."""
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

def gen_frames():
    """Generates webcam frames with game overlay."""
    global player1_score, player2_score, final_winner, round_done

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam access failed!")
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    ai_move = None  

    while True:
        success, frame = cap.read()
        if not success:
            break  

        frame = cv2.resize(frame, (800, 600))

        if final_winner:
            # ✅ Show winner on-screen in the center
            cv2.putText(frame, final_winner, (220, 300), font, 1.5, (0, 0, 255), 3)
        else:
            results = model(frame)[0]
            player1_move = None
            player2_move = None

            # Detect gestures
            for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                label = class_names[int(cls)]
                center_x = (x1 + x2) // 2

                if center_x < frame.shape[1] // 2:
                    player1_move = label
                else:
                    player2_move = label

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), font, 0.8, (0, 255, 0), 2)

            # AI Move in PvC Mode
            if game_mode == "computer" and player1_move:
                if not round_done:
                    ai_move = random.choice(class_names)
                    play_sound("move")
                player2_move = ai_move  
                # ✅ Move AI move display to top middle section in black
                cv2.putText(frame, f"AI Move: {ai_move}", (330, 50), font, 1.2, (0, 0, 0), 3)

            # Determine winner
            if player1_move and player2_move and not round_done:
                winner = get_winner(player1_move, player2_move)
                if winner == "Player 1":
                    player1_score += 1
                elif winner == "Player 2":
                    player2_score += 1
                round_done = True
                save_winning_frame(frame)
                play_sound("win")

            if not player1_move and not player2_move:
                round_done = False

            # Show Scores
            cv2.putText(frame, f"Player 1: {player1_score}", (10, 60), font, 0.9, (255, 255, 0), 2)
            cv2.putText(frame, f"{'Computer' if game_mode == 'computer' else 'Player 2'}: {player2_score}", (10, 90), font, 0.9, (255, 0, 255), 2)

            # ✅ STOP GAME EXACTLY WHEN A PLAYER REACHES 3, 5, or 7
            if player1_score == best_of or player2_score == best_of:
                if player1_score > player2_score:
                    final_winner = "🏆 Winner: Player 1"
                elif game_mode == "computer":
                    final_winner = "🤖 Winner: AI"
                else:
                    final_winner = "🏆 Winner: Player 2"
                break  # ✅ Stop the loop immediately

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

if __name__ == "__main__":
    app.run(debug=True)
