import pandas as pd
import numpy as np
import os

# Create the data directory so it doesn't crash on your computer
os.makedirs("data", exist_ok=True)

def generate_vpp_dataset(
    scenario_name: str, 
    max_steps: int = 100, # 96 steps for the day + 4 extra for the AI's forecast window
    peak_solar_kw: float = 5.0, # Per house
    solar_noise: float = 0.1, 
    base_load_kw: float = 1.5, # Per house
    demand_spike_start: int = None, 
    demand_spike_end: int = None, 
    demand_mult: float = 1.0,
    base_price_mwh: float = 40.0, 
    price_spike_start: int = None, 
    price_spike_end: int = None, 
    price_mult: float = 1.0,
    freq_drop_start: int = None,
    freq_drop_end: int = None
):
    
    data = []
    
    for step in range(max_steps):
        # 1. Time string generation (15-min intervals)
        total_minutes = step * 15
        hours = (total_minutes // 60) % 24
        minutes = total_minutes % 60
        time_str = f"{hours:02d}:{minutes:02d}"
        
        # 2. Solar Generation (Bell curve from ~6 AM to 6 PM)
        solar_kw = 0.0
        if 24 <= step <= 72:
            sun_angle = np.pi * ((step - 24) / 48.0) 
            pure_solar = np.sin(sun_angle) * peak_solar_kw
            noise = np.random.normal(1.0, solar_noise)
            solar_kw = max(0.0, pure_solar * noise)
            
        # 3. Base House Load (The Evening Duck Curve)
        # Higher usage from 5 PM (Step 68) to 9 PM (Step 84)
        evening_bump = 2.0 if 68 <= step <= 84 else 0.0
        load_kw = base_load_kw + evening_bump + np.random.normal(0, 0.2)
        
        if demand_spike_start and demand_spike_end and (demand_spike_start <= step <= demand_spike_end):
            load_kw *= demand_mult
            
        # 4. Wholesale Market Price ($/MWh)
        price_mwh = base_price_mwh + (20.0 if 68 <= step <= 84 else 0.0) + np.random.normal(0, 2.0)
        
        if price_spike_start and price_spike_end and (price_spike_start <= step <= price_spike_end):
            price_mwh *= price_mult
            
        # 5. Grid Physics (Frequency & Voltage)
        grid_freq_hz = 50.0 + np.random.normal(0, 0.02) # Normal Indian/Euro Grid is 50Hz
        grid_voltage_v = 230.0 + (solar_kw * 1.5) - (load_kw * 2.0) + np.random.normal(0, 1.0)
        
        # Apply Grid Stress (Frequency drop)
        if freq_drop_start and freq_drop_end and (freq_drop_start <= step <= freq_drop_end):
            grid_freq_hz -= 0.3 # Drops to ~49.7Hz (Critical Danger!)
            
        # Append the row
        data.append({
            "step": step,
            "time_of_day": time_str,
            "solar_intensity_kw": round(solar_kw, 3),
            "base_house_load_kw": round(max(0.1, load_kw), 3),
            "market_price_per_mwh": round(max(5.0, price_mwh), 2),
            "grid_frequency_hz": round(grid_freq_hz, 3),
            "grid_voltage_v": round(grid_voltage_v, 2)
        })
        
    df = pd.DataFrame(data)
    filepath = f"data/{scenario_name}.csv"
    df.to_csv(filepath, index=False)
    print(f"✅ Generated {filepath} ({len(df)} rows)")

# ==========================================
# GENERATE THE 3 OFFICIAL TASKS
# ==========================================
print("⚡ Booting up VPP Dataset Generator...")

# TASK 1: Easy Spring Day
# Lots of sun, normal loads, safe 50.0Hz grid.
generate_vpp_dataset(
    "easy_spring", 
    peak_solar_kw=6.0, solar_noise=0.05, 
    base_load_kw=1.0, 
    base_price_mwh=45.0
)

# TASK 2: Medium Heatwave
# AC spike starts at 14:00 (Step 56). Grid freq drops to 49.7Hz.
generate_vpp_dataset(
    "medium_heatwave", 
    peak_solar_kw=5.5, solar_noise=0.1, 
    base_load_kw=1.5, 
    demand_spike_start=56, demand_spike_end=72, demand_mult=3.0, 
    base_price_mwh=60.0,
    freq_drop_start=56, freq_drop_end=72
)

# TASK 3: Hard Grid Stress
# Low winter sun. At 18:00 (Step 72), massive 10x price spike ($500+/MWh) for 1 hour.
generate_vpp_dataset(
    "hard_grid_stress", 
    peak_solar_kw=2.0, solar_noise=0.4, # Cloudy/Jumpy solar
    base_load_kw=2.5, 
    base_price_mwh=50.0,
    price_spike_start=72, price_spike_end=76, price_mult=10.0,
    freq_drop_start=72, freq_drop_end=76
)

print("🎉 DONE! Check the 'data' folder. Hand these to Developer 2.")