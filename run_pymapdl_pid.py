import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from ansys.mapdl.core import Mapdl, launch_mapdl

# ==============================================================================
# 1. CONNECT TO RUNNING MAPDL (PORT 50052) OR LAUNCH AUTOMATICALLY
# ==============================================================================
run_dir = os.path.abspath("./mapdl_run")
os.makedirs(run_dir, exist_ok=True)

try:
    print("Attempting to connect to existing MAPDL gRPC server on 127.0.0.1:50052...")
    mapdl = Mapdl(ip="127.0.0.1", port=50052, timeout=10)
    print("Connected to active MAPDL instance!")
except Exception:
    print("No active instance on port 50052. Launching ANSYS252.exe with -smp...")
    ansys_exe = r"C:\Program Files\ANSYS Inc\v252\ansys\bin\winx64\ANSYS252.exe"
    mapdl = launch_mapdl(
        exec_file=ansys_exe,
        run_location=run_dir,
        nproc=8,
        additional_switches="-smp",
        override=True,
        start_timeout=60
    )
    print("Successfully launched and connected to MAPDL!")

# ==============================================================================
# 2. CLEAN & LOAD hotplate_mesh.dat (KEEPING ALL CMBLOCKs & BOUNDARY LOADS)
# ==============================================================================
mapdl.clear()
print("Preparing hotplate_mesh.dat (keeping CMBLOCKs and loads, stripping SOLVE)...")

with open("hotplate_mesh.dat", "r", errors="ignore") as f:
    dat_lines = f.readlines()

prep_lines = []
for line in dat_lines:
    stripped = line.strip().upper()
    # Strip only the automatic SOLVE command so Python controls time-stepping
    if stripped == "SOLVE" or stripped.startswith("SOLVE,"):
        continue
    prep_lines.append(line)

mesh_inp_path = os.path.join(run_dir, "mesh_with_loads.inp")
with open(mesh_inp_path, "w") as f:
    f.writelines(prep_lines)

print("Loading mesh, Named Selections, and boundary conditions into MAPDL...")
mapdl.input(mesh_inp_path)

# ==============================================================================
# 3. ENSURE TRANSIENT THERMAL PROPERTIES (DENSITY & SPECIFIC HEAT) EXIST
# ==============================================================================
mapdl.prep7()

# Ensure Specific Heat (C) and Density (DENS) are defined for all 3 materials
# in case hotplate_mesh.dat was exported from a Steady-State Thermal block:
mapdl.mp("DENS", 1, 2700.0)  # Mat 1: Aluminium 6061 [kg/m^3]
mapdl.mp("C",    1, 960.0)   # Mat 1: Specific Heat [J/(kg*K)]
mapdl.mp("DENS", 2, 1.0)     # Mat 2: Air Gap [kg/m^3]
mapdl.mp("C",    2, 1005.0)
mapdl.mp("DENS", 3, 2330.0)  # Mat 3: Silicon Wafer [kg/m^3]
mapdl.mp("C",    3, 710.0)

# ==============================================================================
# 4. RESOLVE NAMED SELECTION NODE IDs & WAFER BOOLEAN MASK
# ==============================================================================
mapdl.allsel()
all_nnum = mapdl.mesh.nnum

# Grab representative node IDs for RTD-C and RTD-E
mapdl.cmsel("S", "NS_RTD_C")
rtd_c_nodes = mapdl.mesh.nnum
if len(rtd_c_nodes) == 0:
    raise RuntimeError("NS_RTD_C has 0 nodes! Check CMBLOCK name in hotplate_mesh.dat.")
node_rtd_c = int(rtd_c_nodes[0])

mapdl.cmsel("S", "NS_RTD_E")
rtd_e_nodes = mapdl.mesh.nnum
if len(rtd_e_nodes) == 0:
    raise RuntimeError("NS_RTD_E has 0 nodes! Check CMBLOCK name in hotplate_mesh.dat.")
node_rtd_e = int(rtd_e_nodes[0])

# Build a fast boolean mask for all nodes on NS_WAFER_TOP (handles non-contiguous IDs up to 271277+)
mapdl.cmsel("S", "NS_WAFER_TOP")
wafer_top_nnum = mapdl.mesh.nnum
wafer_mask = np.isin(all_nnum, wafer_top_nnum)

mapdl.allsel()
print(f"Tracking RTD-C at Node #{node_rtd_c}, RTD-E at Node #{node_rtd_e}")
print(f"Tracking Wafer Top across {np.sum(wafer_mask)} surface nodes.")

# Quarter-symmetry heated surface areas [m^2] used to convert Watts -> Heat Flux [W/m^2]
# (3 cartridge bores of Ø6.35 mm x 80 mm in quarter-sym, or 2 bores x 50 mm; adjust if needed)
A_cart_qtr = 3 * (np.pi * 0.00635 * 0.080)
A_band_qtr = 0.25 * (np.pi * 0.200 * 0.025)

# ==============================================================================
# 5. CONFIGURE TRANSIENT THERMAL SOLVER SETTINGS
# ==============================================================================
mapdl.slashsolu()
mapdl.antype("TRANS")        # Full Transient Thermal Analysis (ANTYPE, 4)
mapdl.trnopt("FULL")
mapdl.lumpm("OFF")
mapdl.tunif(22.0)            # Initial uniform temperature = 22.0 °C
mapdl.timint("ON")           # Enable transient time integration
mapdl.autots("ON")           # Automatic substeps within each control interval
mapdl.kbc(1)                 # Step-changed heater loads per control interval

# Clear any static Heat Flow (F, ..., HEAT) or HFLUX loads exported from Steady-State
mapdl.fdele("ALL", "HEAT")
mapdl.cmsel("S", "NS_CART_FACES")
mapdl.sfdele("ALL", "HFLUX")
mapdl.cmsel("S", "NS_BAND_FACE")
mapdl.sfdele("ALL", "HFLUX")
mapdl.allsel()

# ==============================================================================
# 6. PID & RATIO CONTROLLER PARAMETERS (Design Document Section 6)
# ==============================================================================
P_CART_MAX_FULL = 800.0      # Max full-plate cartridge power [W]
P_BAND_MAX_FULL = 450.0      # Max full-plate rim band power [W]
P_TOT_MAX_FULL  = P_CART_MAX_FULL + P_BAND_MAX_FULL

# SIMC PI Gains (10 s closed-loop time constant from Design Doc Section 6)
Kp = 194.0                   # Proportional gain [W/K] (full-plate basis)
Ti = 46.0                    # Integral time constant [s]
Ki = Kp / Ti                 # Integral gain [W/(K*s)]
Ki_edge_trim = 0.00015       # Slow integral trim on edge ratio f

# Setpoint & Calibration Offset (Design Doc Section 7)
T_wafer_target = 400.0       # Target Wafer Temperature [°C]
calib_offset   = 14.9        # Plate RTD-C to Wafer offset at 400 °C
T_rtd_setpoint = T_wafer_target + calib_offset
f_base         = 0.60        # Initial edge share (set to 0.18 if matching your Ansys BCs)

RAMP_RATE_K_PER_S = 100.0 / 60.0  # 10 K/min software ramp cap

# Time-stepping configuration
dt = 5.0                     # Control / FEA synchronization step [s]
t_total = 1200.0             # 20 minutes total simulation [s]
n_steps = int(t_total / dt)

# Telemetry arrays
time_hist    = np.zeros(n_steps)
sp_hist      = np.zeros(n_steps)
rtd_c_hist   = np.zeros(n_steps)
rtd_e_hist   = np.zeros(n_steps)
wafer_mean   = np.zeros(n_steps)
wafer_ptp    = np.zeros(n_steps)
p_cart_hist  = np.zeros(n_steps)
p_band_hist  = np.zeros(n_steps)
f_ratio_hist = np.zeros(n_steps)

# Initialize controller states
int_error    = 0.0
f_trim       = 0.0
current_sp   = 22.0
T_rtd_c_curr = 22.0
T_rtd_e_curr = 22.0

# ==============================================================================
# 7. CLOSED-LOOP PYTHON <-> ANSYS CO-SIMULATION LOOP
# ==============================================================================
print("Starting closed-loop PyMAPDL Transient PID simulation...")

with open("pymapdl_pid_telemetry.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "Time_s", "Setpoint_C", "RTD_C", "RTD_E",
        "Wafer_Mean_C", "Wafer_PeakToPeak_K",
        "P_Cartridges_W", "P_Band_W", "Edge_Share_f"
    ])

    for step in range(n_steps):
        t_next = (step + 1) * dt

        # 1. Update Ramp-Rate Limited Setpoint (10 K/min)
        if current_sp < T_rtd_setpoint:
            current_sp = min(T_rtd_setpoint, current_sp + RAMP_RATE_K_PER_S * dt)

        # 2. Primary PI Loop on RTD-C (Full-plate wattage basis)
        error_c = current_sp - T_rtd_c_curr
        P_tot_unclamped = Kp * error_c + Ki * int_error
        P_tot_full = float(np.clip(P_tot_unclamped, 0.0, P_TOT_MAX_FULL))

        # Anti-windup: Clamp integrator when output saturates or |error| > 10 K
        if (0.0 < P_tot_unclamped < P_TOT_MAX_FULL) and (abs(error_c) < 10.0):
            int_error += error_c * dt

        # 3. Edge Ratio + Slow Integral Trim on RTD-E
        error_e = T_rtd_c_curr - T_rtd_e_curr
        if abs(T_rtd_setpoint - T_rtd_c_curr) < 5.0:
            f_trim = float(np.clip(f_trim + Ki_edge_trim * error_e * dt, -0.10, 0.10))

        f_active = float(np.clip(f_base + f_trim, 0.05, 0.95))

        # Split into Centre Cartridges and Edge Band (Full Plate Watts)
        P_band_full = float(np.clip(f_active * P_tot_full, 0.0, P_BAND_MAX_FULL))
        P_cart_full = float(np.clip((1.0 - f_active) * P_tot_full, 0.0, P_CART_MAX_FULL))

        # Convert Full-Plate Watts to Quarter-Symmetry Heat Flux [W/m^2]
        q_flux_cart = (0.25 * P_cart_full) / A_cart_qtr
        q_flux_band = (0.25 * P_band_full) / A_band_qtr

        # 4. Apply Updated Heater Fluxes directly to the Node Components
        mapdl.cmsel("S", "NS_CART_FACES")
        mapdl.sf("ALL", "HFLUX", q_flux_cart)

        mapdl.cmsel("S", "NS_BAND_FACE")
        mapdl.sf("ALL", "HFLUX", q_flux_band)
        mapdl.allsel()

        # 5. Advance the 3D Ansys Transient Solver to t = t_next
        mapdl.time(t_next)
        mapdl.deltim(dt, dt / 2.0, dt)
        mapdl.solve()

        # 6. Read Live 3D Temperatures Back into Python
        T_rtd_c_curr = mapdl.get_value("NODE", node_rtd_c, "TEMP")
        T_rtd_e_curr = mapdl.get_value("NODE", node_rtd_e, "TEMP")

        all_temps = mapdl.post_processing.nodal_temperature()
        w_temps = all_temps[wafer_mask]

        T_w_mean = float(np.mean(w_temps))
        T_w_ptp  = float(np.max(w_temps) - np.min(w_temps))

        # 7. Store & Log Telemetry
        time_hist[step]    = t_next
        sp_hist[step]      = current_sp
        rtd_c_hist[step]   = T_rtd_c_curr
        rtd_e_hist[step]   = T_rtd_e_curr
        wafer_mean[step]   = T_w_mean
        wafer_ptp[step]    = T_w_ptp
        p_cart_hist[step]  = P_cart_full
        p_band_hist[step]  = P_band_full
        f_ratio_hist[step] = f_active

        writer.writerow([
            f"{t_next:.1f}", f"{current_sp:.2f}", f"{T_rtd_c_curr:.3f}", f"{T_rtd_e_curr:.3f}",
            f"{T_w_mean:.3f}", f"{T_w_ptp:.4f}",
            f"{P_cart_full:.1f}", f"{P_band_full:.1f}", f"{f_active:.4f}"
        ])

        print(
            f"[t = {t_next:6.1f} s] SP: {current_sp:6.2f} °C | "
            f"RTD-C: {T_rtd_c_curr:6.2f} °C | Wafer: {T_w_mean:6.2f} °C "
            f"(PtP: {T_w_ptp:.3f} K) | P_cart: {P_cart_full:5.1f} W | P_band: {P_band_full:5.1f} W"
        )

print("Simulation complete! Telemetry saved to pymapdl_pid_telemetry.csv")

# ==============================================================================
# 8. PLOT 3D FEA CLOSED-LOOP RESULTS
# ==============================================================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

ax1.plot(time_hist / 60.0, sp_hist, "k--", label="Ramped RTD-C Setpoint")
ax1.plot(time_hist / 60.0, rtd_c_hist, "r", label="3D Node: RTD-C")
ax1.plot(time_hist / 60.0, rtd_e_hist, "orange", alpha=0.8, label="3D Node: RTD-E")
ax1.plot(time_hist / 60.0, wafer_mean, "b", label="3D Surface: Wafer Mean")
ax1.set_ylabel("Temperature (°C)")
ax1.set_title("PyMAPDL 3D Closed-Loop Transient PID Simulation")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(time_hist / 60.0, p_cart_hist, label="Cartridges Power (Full Plate W)")
ax2.plot(time_hist / 60.0, p_band_hist, label="Edge Band Power (Full Plate W)")
ax2.set_ylabel("Power (W)")
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3.plot(time_hist / 60.0, wafer_ptp, "purple", label="3D Wafer Peak-to-Peak Spread (Max - Min)")
ax3.axhline(1.0, color="r", linestyle="--", label="1.0 K Spec Limit (±0.5 °C)")
ax3.set_xlabel("Time (minutes)")
ax3.set_ylabel("Wafer ΔT (K)")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()