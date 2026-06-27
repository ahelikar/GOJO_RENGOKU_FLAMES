from __future__ import annotations
import cv2
import mediapipe as mp
import numpy as np
import random

WIDTH, HEIGHT = 640, 480
MAX_PARTICLES = 3000

particle_pos = np.zeros((MAX_PARTICLES, 2), dtype=np.float32)
particle_vel = np.zeros((MAX_PARTICLES, 2), dtype=np.float32)
particle_color = np.zeros((MAX_PARTICLES, 3), dtype=np.int32)  
particle_life = np.zeros(MAX_PARTICLES, dtype=np.float32)
particle_max_life = np.zeros(MAX_PARTICLES, dtype=np.float32)
particle_hand_owner = np.zeros(MAX_PARTICLES, dtype=np.int32)  
particle_size = np.zeros(MAX_PARTICLES, dtype=np.int32)

def spawn_particle(index, cx, cy, hand_type_flag):
    particle_pos[index] = [cx + random.uniform(-10, 10), cy + random.uniform(-10, 10)]
    
    angle = random.uniform(0, 2 * np.pi)
    speed = random.uniform(80, 260)
    particle_vel[index] = [np.cos(angle) * speed, np.sin(angle) * speed]
    
    lifetime = random.uniform(0.4, 0.8)
    particle_life[index] = lifetime
    particle_max_life[index] = lifetime
    particle_hand_owner[index] = hand_type_flag
    
    particle_size[index] = random.randint(2, 5)
    
    if hand_type_flag == 0:
        particle_color[index] = [255, 0, 130]   
    else:
        particle_color[index] = [0, 150, 255]   

def update_particles(hand_positions, dt=1.0/30.0):
    for i in range(MAX_PARTICLES):
        if particle_life[i] > 0.0:
            owner = particle_hand_owner[i]
            
            if owner in hand_positions:
                tx, ty = hand_positions[owner]
                dx = tx - particle_pos[i][0]
                dy = ty - particle_pos[i][1]
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance > 2:
                    nx = dx / distance
                    ny = dy / distance
                    
                    vx = -ny
                    vy = nx
                    
                    pull_strength = 400.0
                    vortex_strength = 600.0
                    
                    particle_vel[i][0] += (nx * pull_strength + vx * vortex_strength) * dt * 12.0
                    particle_vel[i][1] += (ny * pull_strength + vy * vortex_strength) * dt * 12.0
            
            particle_vel[i][0] += random.uniform(-45, 45)
            particle_vel[i][1] += random.uniform(-45, 45)
            
            particle_vel[i] *= 0.90
            
            particle_pos[i] += particle_vel[i] * dt
            particle_life[i] -= dt

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
particle_ptr = 0

print("Launching Sharp Crystal-Particle Plasma Engine... Press 'q' to close down.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    
    base_layer = cv2.convertScaleAbs(frame, alpha=0.15, beta=0)
    
    purple_layer = np.zeros_like(base_layer)
    yellow_layer = np.zeros_like(base_layer)
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    active_hand_centers = {}

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[idx].classification[0].label
            
            if hand_label == "Right":
                hand_flag = 0  
            else:
                hand_flag = 1  

            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            knuckle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            
            cx = (wrist.x + knuckle.x) / 2.0 * WIDTH
            cy = (wrist.y + knuckle.y) / 2.0 * HEIGHT
            
            active_hand_centers[hand_flag] = (cx, cy)
            
            for _ in range(65):
                spawn_particle(particle_ptr, cx, cy, hand_flag)
                particle_ptr = (particle_ptr + 1) % MAX_PARTICLES

    update_particles(active_hand_centers)
    
    for i in range(MAX_PARTICLES):
        if particle_life[i] > 0.0:
            px, py = int(particle_pos[i][0]), int(particle_pos[i][1])
            
            if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                owner = particle_hand_owner[i]
                radius = particle_size[i]
                
                life_pct = particle_life[i] / particle_max_life[i]
                color = [int(c * life_pct) for c in particle_color[i]]
                
                if owner == 0:
                    cv2.circle(purple_layer, (px, py), radius=radius, color=color, thickness=-1)
                else:
                    cv2.circle(yellow_layer, (px, py), radius=radius, color=color, thickness=-1)

    for i in range(MAX_PARTICLES):
        if particle_life[i] > 0.55:
            px, py = int(particle_pos[i][0]), int(particle_pos[i][1])
            if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                owner = particle_hand_owner[i]
                if owner == 0:
                    cv2.circle(purple_layer, (px, py), radius=1, color=[255, 255, 255], thickness=-1)
                else:
                    cv2.circle(yellow_layer, (px, py), radius=1, color=[255, 255, 255], thickness=-1)

    combined_particles = cv2.add(purple_layer, yellow_layer)
    final_output = cv2.addWeighted(base_layer, 1.0, combined_particles, 1.9, 0)

    cv2.imshow("Hollow Purple Engine", final_output)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()