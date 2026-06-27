from __future__ import annotations
import cv2
import mediapipe as mp
import numpy as np
import random

WIDTH, HEIGHT = 640, 480
MAX_PARTICLES = 5000

particle_pos = np.zeros((MAX_PARTICLES, 2), dtype=np.float32)
particle_vel = np.zeros((MAX_PARTICLES, 2), dtype=np.float32)
particle_color = np.zeros((MAX_PARTICLES, 3), dtype=np.int32)  
particle_life = np.zeros(MAX_PARTICLES, dtype=np.float32)
particle_max_life = np.zeros(MAX_PARTICLES, dtype=np.float32)
particle_hand_owner = np.zeros(MAX_PARTICLES, dtype=np.int32)  
particle_size = np.zeros(MAX_PARTICLES, dtype=np.int32)

def spawn_particle(index, cx, cy, hand_type_flag, hands_are_mixing=False):
    particle_pos[index] = [cx + random.uniform(-10, 10), cy + random.uniform(-10, 10)]
    
    angle = random.uniform(0, 2 * np.pi)
    
    if hands_are_mixing:
        speed = random.uniform(180, 480)
        particle_life[index] = random.uniform(0.25, 0.55)
        particle_size[index] = random.randint(3, 6)
        
        rng = random.random()
        if rng < 0.55:
            particle_color[index] = [255, 255, 255]
        elif rng < 0.80:
            particle_color[index] = [255, 20, 140]   
        else:
            particle_color[index] = [30, 190, 255]   
    else:
        
        speed = random.uniform(100, 300)
        particle_life[index] = random.uniform(0.4, 0.8)
        particle_size[index] = random.randint(2, 5)
        
        if hand_type_flag == 0:
            particle_color[index] = [255, 0, 120]    
        else:
            particle_color[index] = [0, 150, 255]    
            
    particle_max_life[index] = particle_life[index]
    particle_vel[index] = [np.cos(angle) * speed, np.sin(angle) * speed]
    particle_hand_owner[index] = hand_type_flag

def update_particles(hand_positions, hands_are_mixing=False, mid_x=0, mid_y=0, dt=1.0/30.0):
    for i in range(MAX_PARTICLES):
        if particle_life[i] > 0.0:
            owner = particle_hand_owner[i]
            
            if hands_are_mixing:
                dx = mid_x - particle_pos[i][0]
                dy = mid_y - particle_pos[i][1]
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance > 2:
                    nx = dx / distance
                    ny = dy / distance
                    
                    vx = -ny
                    vy = nx
                    
                    pull_strength = 650.0
                    vortex_strength = 850.0
                    
                    particle_vel[i][0] += (nx * pull_strength + vx * vortex_strength) * dt * 12.0
                    particle_vel[i][1] += (ny * pull_strength + vy * vortex_strength) * dt * 12.0
            
            elif owner in hand_positions:
                tx, ty = hand_positions[owner]
                dx = tx - particle_pos[i][0]
                dy = ty - particle_pos[i][1]
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance > 2:
                    nx = dx / distance
                    ny = dy / distance
                    
                    vx = -ny
                    vy = nx
                    
                    pull_strength = 450.0
                    vortex_strength = 650.0
                    
                    particle_vel[i][0] += (nx * pull_strength + vx * vortex_strength) * dt * 12.0
                    particle_vel[i][1] += (ny * pull_strength + vy * vortex_strength) * dt * 12.0
            
            particle_vel[i][0] += random.uniform(-60, 60)
            particle_vel[i][1] += random.uniform(-60, 60)
            
            particle_vel[i] *= 0.88  
            particle_pos[i] += particle_vel[i] * dt
            particle_life[i] -= dt


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
window_name = "Full Screen Display"






particle_ptr = 0


gas_accumulation_canvas = None

print("Launching Hybrid Particle-Gaseous Plasma Engine... Press 'q' to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    
    if gas_accumulation_canvas is None:
        gas_accumulation_canvas = np.zeros_like(frame)
        
    
    base_layer = cv2.convertScaleAbs(frame, alpha=0.15, beta=0)
    
    
    current_particle_layer = np.zeros_like(frame)
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    active_hand_centers = {}
    hands_are_mixing = False
    mid_x, mid_y = 0, 0

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

        
        if 0 in active_hand_centers and 1 in active_hand_centers:
            lx, ly = active_hand_centers[0]
            rx, ry = active_hand_centers[1]
            dist_between_hands = np.sqrt((lx - rx)**2 + (ly - ry)**2)
            
            if dist_between_hands < 135:
                hands_are_mixing = True
                mid_x = (lx + rx) / 2.0
                mid_y = (ly + ry) / 2.0

        
        for hand_flag, (cx, cy) in active_hand_centers.items():
            spawn_target_x = mid_x if hands_are_mixing else cx
            spawn_target_y = mid_y if hands_are_mixing else cy
            
            for _ in range(85):
                spawn_particle(particle_ptr, spawn_target_x, spawn_target_y, hand_flag, hands_are_mixing)
                particle_ptr = (particle_ptr + 1) % MAX_PARTICLES

    
    update_particles(active_hand_centers, hands_are_mixing, mid_x, mid_y)
    
    
    for i in range(MAX_PARTICLES):
        if particle_life[i] > 0.0:
            px, py = int(particle_pos[i][0]), int(particle_pos[i][1])
            
            if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                radius = particle_size[i]
                life_pct = particle_life[i] / particle_max_life[i]
                color = [int(c * life_pct) for c in particle_color[i]]
                
                cv2.circle(current_particle_layer, (px, py), radius=radius, color=color, thickness=-1)

    #
    gas_accumulation_canvas = cv2.convertScaleAbs(gas_accumulation_canvas, alpha=0.78, beta=0)
    
    hazy_smoke_layer = cv2.GaussianBlur(current_particle_layer, (15, 15), 0)
    
    gas_accumulation_canvas = cv2.add(gas_accumulation_canvas, hazy_smoke_layer)
    
    combined_plasma_fx = cv2.addWeighted(gas_accumulation_canvas, 1.2, current_particle_layer, 2.0, 0)

    if hands_are_mixing:
        cv2.circle(combined_plasma_fx, (int(mid_x), int(mid_y)), radius=16, color=[255, 255, 255], thickness=-1)
        
        for i in range(MAX_PARTICLES):
            if particle_life[i] > 0.48 and random.random() < 0.4:
                px, py = int(particle_pos[i][0]), int(particle_pos[i][1])
                if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                    cv2.circle(combined_plasma_fx, (px, py), radius=1, color=[255, 255, 255], thickness=-1)

    final_output = cv2.addWeighted(base_layer, 1.0, combined_plasma_fx, 1.8, 0)

    cv2.imshow("Hollow Purple Engine", final_output)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()