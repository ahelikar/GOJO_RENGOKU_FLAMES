# GOJO_RENGOKU_FLAMES
An interactive, real-time computer vision application that generates a swirling plasma vortex of particles orbiting your hands. Built using MediaPipe for high-fidelity hand tracking and OpenCV for dynamic, high-performance particle physics rendering.

🚀 Features
Real-Time Hand Tracking: Leverages Google MediaPipe Hands to detect and track hand positions instantly through your webcam.

Dual-Hand Color Mapping: Emits distinctive colored particles depending on which hand is detected (Pink/Magenta for the Right hand, Cyan/Blue for the Left hand).

Vortex Physics Engine: Simulates dynamic gravitational pull and rotational vortex strengths, forcing particles to realistically spiral around the centers of your palms.

Particle Lifecycle & Fading: Features a custom particle management loop where elements dynamically change size, fade out based on their remaining life, and emit a bright "core" glow when newly spawned.

🛠️ Tech Stack & Prerequisites
Python 3.8+

OpenCV (opencv-python): Handles video capture, frame matrix manipulation, and hardware-accelerated circle rendering.

MediaPipe: Provides the machine learning pipeline for robust landmark tracking.

NumPy: Powers the vectorized math calculations for 3,000+ simultaneous particles.

📦 Installation
Clone the repository:

Bash
git clone https://github.com/your-username/hollow-purple-engine.git
cd hollow-purple-engine
Install the required dependencies:

Bash
pip install opencv-python mediapipe numpy
💻 Usage
Run the main Python script to initialize your webcam and launch the particle engine:

Bash
python main.py
Controls & Navigation
Interact: Move your hands in front of your webcam. Watch the particle stream spawn from your palms and orbit your movements.

Exit: Press the q key on your keyboard while focusing on the display window to safely shut down the engine.

🔬 How It Works Under the Hood
[Webcam Feed] ──> [MediaPipe Hand Detection] ──> [Calculate Hand Center]
                                                          │
                                                          ▼
[Render Output] <── [Vectorized Physics Updates] <── [Spawn Particles]
Center Calculation: The engine calculates a steady tracking anchor point midway between your WRIST and MIDDLE_FINGER_MCP (knuckle) landmarks.

Velocity Vector Field: Every active particle experiences a continuous update phase pulling it toward the hand center (pull_strength) combined with a perpendicular orbital velocity (vortex_strength).

Layer Blending: The webcam feed is dimmed down to create a dark environment. The particles are drawn onto separate color layers and merged back onto the dimmed background using an additive cv2.addWeighted pass for a high-intensity glow effect.

📄 License
This project is open-source and available under the MIT License
