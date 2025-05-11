import math
import random
import numpy as np

# Attempt GPU acceleration with CuPy
try:
    import cupy as cp
    xp = cp
    print("[INFO] Using CuPy for GPU acceleration.")
except ImportError:
    # Fallback to NumPy if CuPy is not available
    xp = np
    print("[WARNING] CuPy not available; using NumPy CPU fallback.")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

##############################################################################
# ENGINEERING THOUGHT MODULAR COMPOSITION (CRT-like Implementation)
##############################################################################
def engineering_thought_modular_composition(residues, moduli):
    """
    CRT-style engineering metric combining multi-parameter sets into 
    a single 'delta-V' or performance measure. Demonstrates how modular
    thought can combine relevant parameters into scenario scoring.
    """
    from math import prod, log
    M = prod(moduli)
    print(f"[DEBUG] engineering_thought_modular_composition called with residues={residues}, moduli={moduli}")
    print(f"[DEBUG] Computed product of moduli M={M}")

    if len(residues) == 2 and len(moduli) == 2:
        mi, mf = residues
        isp, g = moduli
        if mi <= mf:
            print("[DEBUG] mi <= mf, returning 0")
            return 0
        dv = isp * g * log(mi / mf)
        result = dv + 0.1 * dv**0.8 + M
        print(f"[DEBUG] Returning {result} for two-parameter model")
        return result

    elif len(residues) == 3 and len(moduli) == 3:
        mi, thrust, isp = residues
        mf, burn_time, g = moduli
        if mi <= mf:
            print("[DEBUG] mi <= mf, returning 0")
            return 0
        dv = isp * g * log(mi / mf)
        avg_acc = thrust / ((mi + mf) / 2)
        result = dv + 0.2 * avg_acc**0.7 + burn_time**0.3 + M
        print(f"[DEBUG] Returning {result} for three-parameter model")
        return result

    print("[DEBUG] No matching conditions, returning 0")
    return 0

##############################################################################
# DATA PIPELINES (SIGNATURES, ETC.)
##############################################################################
class SignatureDataset(Dataset):
    """
    Demonstration dataset for loading / storing target signature data.
    This could handle RCS, IR, or other attributes. In practice, you'd
    store or generate these from real or synthetic tables, images, etc.
    """
    def __init__(self, size=1000):
        # For demonstration, randomly generate "signature data"
        self.size = size
        self.data = []
        for _ in range(size):
            # Synthetic example: [aspect_angle, freq_band, rcs_value]
            aspect_angle = random.uniform(0, 180)
            freq_band = random.choice(["X", "Ku", "C"])
            rcs_value = random.uniform(0.01, 5.0)
            self.data.append((aspect_angle, freq_band, rcs_value))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

def load_signature_data(batch_size=32, dataset_size=1000):
    dataset = SignatureDataset(size=dataset_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

##############################################################################
# SENSOR MODULES
##############################################################################
class RadarSensor:
    def __init__(self, freq="X", noise_level=0.01, jam_resistance=1.0):
        self.freq = freq
        self.noise_level = noise_level
        self.jam_resistance = jam_resistance
        self.confidence = 1.0

    def sense(self, missile_state, target_state):
        """
        missile_state, target_state: dicts containing position, velocity, signature info, etc.
        returns a dict representing radar observation
        """
        # naive range calculation
        dx = target_state["pos"][0] - missile_state["pos"][0]
        dy = target_state["pos"][1] - missile_state["pos"][1]
        dz = target_state["pos"][2] - missile_state["pos"][2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        # Suppose RCS from target signature
        rcs = target_state["rcs"]
        # simple radar equation (extremely simplified):
        # received_power ~ rcs / (dist^4) + noise
        if dist < 1e-3:
            dist = 1e-3
        rp = rcs / (dist**4) + random.gauss(0, self.noise_level)
        
        # simulate jamming effects
        jam_factor = target_state.get("jam_strength", 0.0)
        # effective jam could degrade reading
        # degrade confidence if jam_factor is large
        effective_jam = jam_factor / self.jam_resistance
        self.confidence = max(0.0, 1.0 - effective_jam)

        return {
            "range": dist,
            "power": rp * self.confidence,
            "confidence": self.confidence,
            "freq": self.freq,
        }

class IRSensor:
    def __init__(self, noise_level=0.01):
        self.noise_level = noise_level
        self.confidence = 1.0

    def sense(self, missile_state, target_state):
        dx = target_state["pos"][0] - missile_state["pos"][0]
        dy = target_state["pos"][1] - missile_state["pos"][1]
        dz = target_state["pos"][2] - missile_state["pos"][2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        # Let's define IR signature
        ir_sig = target_state.get("ir_signature", 1.0)
        # IR intensity ~ ir_sig / (dist^2) plus some noise
        if dist < 1e-3:
            dist = 1e-3
        intensity = ir_sig / (dist**2)
        intensity += random.gauss(0, self.noise_level*intensity)

        # flares could trick the sensor
        # if flares are deployed, either degrade or create false contact
        # For simplicity, assume target has an attribute "flares_active" with some probability
        flares = target_state.get("flares_active", False)
        if flares:
            # random chance IR is confused
            confusion_factor = random.random()
            if confusion_factor > 0.7:
                intensity *= 0.1  # lose track
                self.confidence = 0.3
            else:
                self.confidence = 0.8
        else:
            self.confidence = 1.0

        return {
            "range_approx": dist,  # IR might not precisely measure range
            "intensity": intensity,
            "confidence": self.confidence,
        }

class PassiveRFSensor:
    def __init__(self, sensitivity=1.0):
        self.sensitivity = sensitivity
        self.confidence = 1.0

    def sense(self, missile_state, target_state):
        # If target is emitting radar or jam:
        rf_emit = target_state.get("rf_emit", 0.0)
        jam_strength = target_state.get("jam_strength", 0.0)
        dx = target_state["pos"][0] - missile_state["pos"][0]
        dy = target_state["pos"][1] - missile_state["pos"][1]
        dz = target_state["pos"][2] - missile_state["pos"][2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist < 1e-3:
            dist = 1e-3

        # detection ~ (rf_emit + jam_strength) / dist^2
        detection = (rf_emit + jam_strength) / (dist**2)
        detection *= random.uniform(0.9, 1.1)  # small noise
        if detection > self.sensitivity:
            self.confidence = 1.0
        else:
            self.confidence = detection / self.sensitivity

        return {
            "bearing": (math.atan2(dy, dx), math.atan2(dz, math.sqrt(dx*dx+dy*dy))),
            "confidence": self.confidence,
            "signal_level": detection,
        }

##############################################################################
# SENSOR FUSION MODULE (NEURAL NETWORK)
##############################################################################
class SensorFusionNN(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, output_dim=8):
        """
        input_dim: combined dimension of sensor readings
        hidden_dim: size of hidden layer
        output_dim: fused feature dimension
        """
        super(SensorFusionNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SensorFusionModule:
    def __init__(self, model_path=None):
        # For demonstration, the input dimension is sum of
        # [radar_range, radar_power, radar_conf, IR_range_approx, IR_intensity, IR_conf,
        #   RF_bearing_x2, RF_conf, RF_signal_level] => we might tailor it
        # Just an example dimension
        input_dim = 9
        self.nn_model = SensorFusionNN(input_dim=input_dim, hidden_dim=64, output_dim=8)
        if model_path is not None:
            self.nn_model.load_state_dict(torch.load(model_path))
        self.nn_model.eval()

    def fuse(self, radar_data, ir_data, rf_data):
        # Flatten sensor data into input vector
        # radar_data: {range, power, confidence}
        # ir_data: {range_approx, intensity, confidence}
        # rf_data: {bearing=(az, el), confidence, signal_level}
        az, el = rf_data["bearing"]
        input_vec = [
            radar_data["range"], 
            radar_data["power"],
            radar_data["confidence"],
            ir_data["range_approx"],
            ir_data["intensity"],
            ir_data["confidence"],
            az, el,
            rf_data["confidence"] * rf_data["signal_level"]
        ]
        input_tensor = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            fused_out = self.nn_model(input_tensor)[0].numpy()
        # Suppose fused_out is an 8-dim vector representing [est_x, est_y, est_z, v_x, v_y, v_z, overall_conf, etc...]
        return fused_out

##############################################################################
# SEEKER NEURAL NETWORK (RNN or simpler approach)
##############################################################################
class SeekerRNN(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, output_dim=4):
        """
        input_dim: dimension from fusion output
        hidden_dim: LSTM hidden
        output_dim: e.g. [guidance_cmd_pitch, yaw, roll_rate, confidence]
        """
        super(SeekerRNN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, hidden=None):
        """
        x: (batch_size, seq_len, input_dim)
        hidden: (h0, c0) if provided
        """
        lstm_out, hidden_out = self.lstm(x, hidden)
        # take the last output
        last_out = lstm_out[:, -1, :]
        out = self.fc(self.relu(last_out))
        return out, hidden_out

##############################################################################
# SIMULATION COMPONENTS (MISSILE, TARGET, SCENARIO)
##############################################################################
class MissileState:
    def __init__(self, pos=(0,0,0), vel=(0,0,0), mass=300, prop_force=0):
        self.pos = list(pos)
        self.vel = list(vel)
        self.mass = mass
        self.prop_force = prop_force

    def update(self, dt):
        # Very naive physics: a = F/m in direction of velocity
        # ignoring drag for simplicity
        # you would do a more complete 6DOF for realistic simulation
        speed = math.sqrt(self.vel[0]**2 + self.vel[1]**2 + self.vel[2]**2)
        if speed > 1e-6:
            a_mag = self.prop_force / self.mass
            # accelerate in direction of velocity
            self.vel[0] += a_mag * (self.vel[0]/speed) * dt
            self.vel[1] += a_mag * (self.vel[1]/speed) * dt
            self.vel[2] += a_mag * (self.vel[2]/speed) * dt

        # update position
        self.pos[0] += self.vel[0] * dt
        self.pos[1] += self.vel[1] * dt
        self.pos[2] += self.vel[2] * dt

    def as_dict(self):
        return {
            "pos": tuple(self.pos),
            "vel": tuple(self.vel),
            "mass": self.mass
        }

class TargetState:
    def __init__(self, pos=(10000,0,2000), vel=(-200,0,0), rcs=1.0, ir_signature=1.0,
                 flares_active=False, jam_strength=0.0, rf_emit=0.0):
        self.pos = list(pos)
        self.vel = list(vel)
        self.rcs = rcs
        self.ir_signature = ir_signature
        self.flares_active = flares_active
        self.jam_strength = jam_strength
        self.rf_emit = rf_emit

    def update(self, dt):
        # Simple linear
        self.pos[0] += self.vel[0] * dt
        self.pos[1] += self.vel[1] * dt
        self.pos[2] += self.vel[2] * dt

    def as_dict(self):
        return {
            "pos": tuple(self.pos),
            "vel": tuple(self.vel),
            "rcs": self.rcs,
            "ir_signature": self.ir_signature,
            "flares_active": self.flares_active,
            "jam_strength": self.jam_strength,
            "rf_emit": self.rf_emit
        }

##############################################################################
# MAIN TRAINING LOOP (SKELETON)
##############################################################################
def run_simulation_episode(missile, target, radar, ir, rf, fusion, seeker_rnn, 
                           dt=0.1, max_time=60.0):
    """
    Run one simulation of missile-target engagement.
    Returns logs or final outcome.
    """
    time_elapsed = 0.0
    hidden_state = None
    logs = []

    while time_elapsed < max_time:
        missile.update(dt)
        target.update(dt)

        # gather sensor data
        r_data = radar.sense(missile.as_dict(), target.as_dict())
        i_data = ir.sense(missile.as_dict(), target.as_dict())
        rf_data = rf.sense(missile.as_dict(), target.as_dict())

        # fuse sensor data
        fused_out = fusion.fuse(r_data, i_data, rf_data)
        # shape for RNN
        fused_tensor = torch.tensor(fused_out, dtype=torch.float32).view(1,1,-1) # batch=1, seq=1
        seeker_output, hidden_state = seeker_rnn(fused_tensor, hidden_state)
        seeker_output = seeker_output.detach().numpy()[0]

        # interpret seeker output as acceleration commands or velocity increments
        # e.g. [pitch_acc, yaw_acc, roll_acc, conf]
        # naive approach: let's just set missile.vel based on some commanded direction
        pitch_cmd = seeker_output[0]
        yaw_cmd   = seeker_output[1]
        # ignoring roll
        # we approximate a new direction
        # in real scenario, you'd do advanced 3D geometry
        # here, let's do a hack: adjust missile velocity's x,y by pitch,yaw
        missile.vel[1] += pitch_cmd * dt
        missile.vel[2] += yaw_cmd * dt

        # check for intercept
        dist = math.sqrt((missile.pos[0]-target.pos[0])**2 + 
                         (missile.pos[1]-target.pos[1])**2 +
                         (missile.pos[2]-target.pos[2])**2)
        if dist < 10.0:
            # hit
            return True, logs

        logs.append((time_elapsed, dist, r_data, i_data, rf_data, seeker_output.tolist()))
        time_elapsed += dt
    
    return False, logs

def train_pl15_seeker(num_episodes=50):
    # Initialize modules
    radar = RadarSensor(freq="X", noise_level=0.01, jam_resistance=1.0)
    ir = IRSensor(noise_level=0.01)
    rf = PassiveRFSensor(sensitivity=0.1)
    fusion = SensorFusionModule(model_path=None)
    seeker_rnn = SeekerRNN(input_dim=8, hidden_dim=32, output_dim=4)

    # Basic optimizer
    optimizer = optim.Adam(seeker_rnn.parameters(), lr=1e-3)
    loss_func = nn.MSELoss()

    for episode in range(num_episodes):
        # random scenario init
        missile = MissileState(pos=(0,0,1000), vel=(300,0,0), mass=300, prop_force=500)
        # random jam, flares
        jam_strength = random.uniform(0.0, 2.0)
        flares_active = random.random() < 0.3
        target = TargetState(pos=(10000, random.uniform(-500,500),1000+random.uniform(-200,200)),
                             vel=(-200+random.uniform(-50,50),0,random.uniform(-10,10)),
                             rcs=random.uniform(0.1,3.0),
                             ir_signature=random.uniform(0.5,3.0),
                             flares_active=flares_active,
                             jam_strength=jam_strength,
                             rf_emit=random.uniform(0.0, 2.0))

        # We'll do a naive supervised target approach:
        # we know we want to reduce distance over time, so we'll accumulate a 'loss'
        done, logs = run_simulation_episode(
            missile, target, radar, ir, rf, fusion, seeker_rnn, 
            dt=0.1, max_time=40
        )

        # We'll do a trivial supervised approach: 
        # we assume final distance => 0 is good. 
        # We'll backprop some MSE on last few distances
        # This is obviously not a rigorous RL or stable approach,
        # but just a demonstration
        if len(logs) > 0:
            # compute average dist
            dists = xp.array([log_item[1] for log_item in logs])
            final_dist = dists[-1]
            # define a pseudo loss
            # we want final dist -> 0
            # We'll treat it as if we want the final dist to be 0, so MSE is final_dist^2
            # this is not truly correct, but for demonstration
            distance_loss = final_dist**2
            distance_loss = torch.tensor(distance_loss, requires_grad=True, dtype=torch.float32)
            
            optimizer.zero_grad()
            distance_loss.backward()
            optimizer.step()

        if done:
            print(f"[TRAIN] Episode {episode}: HIT target!")
        else:
            print(f"[TRAIN] Episode {episode}: MISS. Final dist ~ {final_dist:.2f}")

    # Save final model
    torch.save(seeker_rnn.state_dict(), "pl15_seeker_rnn_final.pt")
    return seeker_rnn

##############################################################################
# MAIN EXECUTION
##############################################################################
if __name__ == "__main__":
    # Use the engineering_thought_modular_composition
    test1 = engineering_thought_modular_composition((549_054, 25_600), (348, 9.80665))
    test2 = engineering_thought_modular_composition((120_000, 934_000, 450), (40_000, 360, 9.80665))
    print(f"Test1: {test1:.6f}, Test2: {test2:.6f}")

    # Train the PL-15 "like" system
    pl15_model = train_pl15_seeker(num_episodes=10)
    print("[MAIN] Training completed. Model saved as pl15_seeker_rnn_final.pt")
