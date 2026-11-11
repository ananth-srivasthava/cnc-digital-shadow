import numpy as np
import pandas as pd
import simpy
from scipy import stats
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================================================================
# PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="CNC Manufacturing Cell DES — Fixed vs. Condition-Aware Dispatching",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    .spec-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #1e293b;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    .spec-title { font-weight: 700; font-size: 1.0rem; color: #0f172a; margin-bottom: 4px; }
    .spec-body { font-size: 0.92rem; color: #334155; line-height: 1.45; }
    </style>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# 1. COMMON RANDOM NUMBERS (CRN) PRE-GENERATOR
# ==============================================================================
def generate_crn_inputs(num_jobs: int, shift_minutes: float, weibull_beta: float, weibull_eta: float, seed: int):
    """
    Pre-generates all stochastic job and machine failure parameters using independent,
    indexed numpy random streams so Experiment A (Fixed Alternating) and Experiment B
    (Condition-Aware Dispatching) experience identical common random numbers (CRN).
    """
    rng = np.random.default_rng(seed)

    # Mean arrival rate scaled so ~200 jobs arrive across ~430 minutes of the 480-min shift
    mean_interarrival = (shift_minutes * 0.88) / num_jobs
    interarrivals = rng.exponential(scale=mean_interarrival, size=num_jobs)
    arrival_times = np.cumsum(interarrivals)

    # CNC Turning nominal processing times [min] (Log-normal / Gamma-like around 3.6 min)
    cnc_proc_times = np.clip(rng.normal(loc=3.6, scale=0.65, size=num_jobs), 1.8, 6.0)
    # Tool wear increment per job (normalized units, ~28-35 jobs per insert life)
    wear_increments = np.clip(rng.normal(loc=0.032, scale=0.006, size=num_jobs), 0.015, 0.055)
    # Thermal load increment per job (normalized units)
    thermal_increments = np.clip(rng.normal(loc=0.075, scale=0.012, size=num_jobs), 0.04, 0.12)

    # Downstream station times [min]
    aoi_proc_times = np.clip(rng.normal(loc=1.35, scale=0.20, size=num_jobs), 0.8, 2.2)
    cobot_proc_times = np.clip(rng.normal(loc=1.25, scale=0.15, size=num_jobs), 0.8, 2.0)

    # Uniform[0,1] rolls for AOI quality check (paired per job)
    aoi_quality_rolls = rng.uniform(0.0, 1.0, size=num_jobs)

    # Pre-generate Weibull effective-age thresholds & repair durations for up to 25 failures per CNC
    # Weibull inverse CDF: L = eta * (-ln(U)) ** (1 / beta)
    u_draws = rng.uniform(0.001, 0.999, size=(2, 25))
    weibull_thresholds = weibull_eta * ((-np.log(u_draws)) ** (1.0 / weibull_beta))
    unplanned_repair_times = np.clip(rng.gamma(shape=4.0, scale=5.5, size=(2, 25)), 10.0, 45.0)
    preventive_maint_times = np.clip(rng.normal(loc=7.5, scale=1.0, size=(2, 25)), 5.0, 11.0)

    jobs_table = pd.DataFrame({
        "job_id": np.arange(1, num_jobs + 1),
        "arrival_time": arrival_times,
        "cnc_proc_time": cnc_proc_times,
        "wear_inc": wear_increments,
        "thermal_inc": thermal_increments,
        "aoi_proc_time": aoi_proc_times,
        "cobot_proc_time": cobot_proc_times,
        "quality_roll": aoi_quality_rolls
    })

    machine_crn = {
        "weibull_thresholds": weibull_thresholds,
        "unplanned_repair_times": unplanned_repair_times,
        "preventive_maint_times": preventive_maint_times
    }
    return jobs_table, machine_crn


# ==============================================================================
# 2. SIMPY DISCRETE-EVENT MANUFACTURING CELL ENGINE
# ==============================================================================
class CNCTurningCenter:
    def __init__(self, env: simpy.Environment, cnc_id: int, machine_crn: dict, cooling_rate: float = 0.035):
        self.env = env
        self.cnc_id = cnc_id  # 0 for CNC 1, 1 for CNC 2
        self.name = f"CNC {cnc_id + 1}"
        self.resource = simpy.Resource(env, capacity=1)

        # Condition state variables
        self.wear_index = 0.0          # W_i in [0, 1+]
        self.thermal_index = 0.15      # Theta_i in [0.15, 1+]
        self.effective_age = 0.0       # Cumulative degradation-weighted operating minutes
        self.last_state_update = 0.0
        self.cooling_rate = cooling_rate

        # CRN failure tables
        self.weibull_Pool = machine_crn["weibull_thresholds"][cnc_id]
        self.repair_Pool = machine_crn["unplanned_repair_times"][cnc_id]
        self.pm_Pool = machine_crn["preventive_maint_times"][cnc_id]
        self.fail_counter = 0
        self.pm_counter = 0

        # Real-time tracking for dispatching & KPIs
        self.is_under_repair = False
        self.repair_end_time = 0.0
        self.current_job_end_time = 0.0
        self.queued_work_time = 0.0

        # Cumulative KPI counters
        self.busy_time = 0.0
        self.downtime = 0.0
        self.unplanned_breakdowns = 0
        self.preventive_replacements = 0
        self.parts_completed = 0
        self.energy_kwh = 0.0

    def update_Cooling(self):
        """Newton's law of cooling toward ambient baseline (0.15) during elapsed time."""
        dt = max(self.env.now - self.last_state_update, 0.0)
        if dt > 0:
            self.thermal_index = 0.15 + (self.thermal_index - 0.15) * np.exp(-self.cooling_rate * dt)
            self.last_state_update = self.env.now

    def estimated_completion_time(self, new_job_proc_time: float) -> float:
        """Calculates T_hat_i: expected wait + processing time if assigned a new job now."""
        self.update_Cooling()
        remaining_active = max(self.current_job_end_time - self.env.now, 0.0)
        remaining_repair = max(self.repair_end_time - self.env.now, 0.0) if self.is_under_repair else 0.0
        return remaining_active + remaining_repair + self.queued_work_time + new_job_proc_time


def run_cell_simulation(
    strategy: str,
    jobs_df: pd.DataFrame,
    machine_crn: dict,
    shift_minutes: float = 480.0,
    alpha_wear: float = 12.0,
    beta_thermal: float = 10.0,
    pm_wear_threshold: float = 0.90
):
    """
    Executes one complete 8-hour shift simulation of the 5-stage manufacturing cell:
    1. Raw Material Buffer -> 2. Parallel CNC 1 & CNC 2 -> 3. AOI Station -> 4. Cobot Palletizer -> 5. Outlet
    strategy: 'fixed' (Experiment A) or 'smart' (Experiment B)
    """
    env = simpy.Environment()

    cnc_machines = [
        CNCTurningCenter(env, 0, machine_crn),
        CNCTurningCenter(env, 1, machine_crn)
    ]
    aoi_station = simpy.Resource(env, capacity=1)
    cobot_station = simpy.Resource(env, capacity=1)

    gantt_logs = []
    telemetry_logs = []
    job_records = []
    wip_trace = []

    active_wip = 0
    good_parts_outlet = 0
    scrapped_parts = 0

    def log_gantt(station_name: str, state: str, start_t: float, end_t: float, job_id_str: str):
        clipped_start = min(start_t, shift_minutes)
        clipped_end = min(end_t, shift_minutes)
        if clipped_end > clipped_start + 1e-4:
            gantt_logs.append({
                "Station": station_name,
                "State": state,
                "Start": clipped_start,
                "Finish": clipped_end,
                "Duration": clipped_end - clipped_start,
                "Details": job_id_str
            })

    def dispatch_job(job_row) -> CNCTurningCenter:
        j_id = int(job_row["job_id"])
        proc_t = float(job_row["cnc_proc_time"])

        if strategy == "fixed":
            # Experiment A: Strict alternating allocation (Odd -> CNC 1, Even -> CNC 2)
            chosen_idx = (j_id - 1) % 2
            return cnc_machines[chosen_idx]
        else:
            # Experiment B: Condition-Aware Dynamic Dispatching
            # S_i = T_hat_i + alpha * W_i + beta * Theta_i + Penalty_ineligible
            scores = []
            for m in cnc_machines:
                m.update_Cooling()
                t_hat = m.estimated_completion_time(proc_t)
                penalty = 0.0
                if m.is_under_repair:
                    penalty += 25.0
                if m.wear_index >= pm_wear_threshold or m.thermal_index >= 0.92:
                    penalty += 15.0
                score = t_hat + (alpha_wear * m.wear_index) + (beta_thermal * m.thermal_index) + penalty
                scores.append(score)

            chosen_idx = int(np.argmin(scores))
            return cnc_machines[chosen_idx]

    def process_job(job_row):
        nonlocal active_wip, good_parts_outlet, scrapped_parts

        j_id = int(job_row["job_id"])
        arr_t = float(job_row["arrival_time"])
        cnc_nom_t = float(job_row["cnc_proc_time"])
        w_inc = float(job_row["wear_inc"])
        th_inc = float(job_row["thermal_inc"])
        aoi_t = float(job_row["aoi_proc_time"])
        cobot_t = float(job_row["cobot_proc_time"])
        q_roll = float(job_row["quality_roll"])

        # Wait until arrival
        yield env.timeout(arr_t)
        if env.now >= shift_minutes:
            return

        active_wip += 1
        wip_trace.append((env.now, active_wip))

        # ----------------------------------------------------------------------
        # STAGE 1 & 2: RAW MATERIAL BUFFER & PARALLEL CNC TURNING CENTERS
        # ----------------------------------------------------------------------
        target_cnc = dispatch_job(job_row)
        target_cnc.queued_work_time += cnc_nom_t
        queue_enter_t = env.now

        with target_cnc.resource.request() as req:
            yield req
            wait_cnc = env.now - queue_enter_t
            target_cnc.queued_work_time = max(target_cnc.queued_work_time - cnc_nom_t, 0.0)

            if env.now >= shift_minutes:
                active_wip -= 1
                return

            target_cnc.update_Cooling()

            #In-cycle Preventive Tool Replacement / Cooldown check (Experiment B proactive threshold)
            if strategy == "smart" and (target_cnc.wear_index >= pm_wear_threshold or target_cnc.thermal_index >= 0.92):
                pm_dur = float(target_cnc.pm_Pool[target_cnc.pm_counter % 25])
                target_cnc.pm_counter += 1
                target_cnc.preventive_replacements += 1
                target_cnc.is_under_repair = True
                target_cnc.repair_end_time = env.now + pm_dur
                pm_start = env.now
                yield env.timeout(pm_dur)
                log_gantt(target_cnc.name, "Preventive Tool Change", pm_start, env.now, f"PM before Job {j_id}")
                target_cnc.downtime += min(env.now, shift_minutes) - min(pm_start, shift_minutes)
                target_cnc.wear_index = 0.0
                target_cnc.thermal_index = 0.20
                target_cnc.effective_age *= 0.25  # Fresh insert resets effective wear-out hazard age
                target_cnc.last_state_update = env.now
                target_cnc.is_under_repair = False

            # Actual cutting time increases slightly with dull tool wear (up to +12%)
            actual_cut_t = cnc_nom_t * (1.0 + 0.12 * target_cnc.wear_index)
            cut_start = env.now
            target_cnc.current_job_end_time = cut_start + actual_cut_t

            # Advance degradation-accelerated effective age
            accel_factor = 1.0 + 1.8 * (target_cnc.wear_index ** 2) + 1.5 * (target_cnc.thermal_index ** 2)
            next_eff_age = target_cnc.effective_age + actual_cut_t * accel_factor
            weibull_limit = float(target_cnc.weibull_Pool[target_cnc.fail_counter % 25])

            # Check if machine suffers a Weibull breakdown or over-wear failure mid-cut
            if next_eff_age >= weibull_limit or target_cnc.wear_index + w_inc >= 1.08 or target_cnc.thermal_index + th_inc >= 1.08:
                # Partial cut before breakdown
                partial_t = actual_cut_t * 0.5
                yield env.timeout(partial_t)
                log_gantt(target_cnc.name, "Processing", cut_start, env.now, f"Job {j_id} (Interrupted)")
                target_cnc.busy_time += max(min(env.now, shift_minutes) - min(cut_start, shift_minutes), 0.0)

                # Unplanned Breakdown & Repair
                rep_dur = float(target_cnc.repair_Pool[target_cnc.fail_counter % 25])
                target_cnc.fail_counter += 1
                target_cnc.unplanned_breakdowns += 1
                target_cnc.is_under_repair = True
                target_cnc.repair_end_time = env.now + rep_dur
                rep_start = env.now
                yield env.timeout(rep_dur)
                log_gantt(target_cnc.name, "Unplanned Breakdown", rep_start, env.now, f"Repair on Job {j_id}")
                target_cnc.downtime += max(min(env.now, shift_minutes) - min(rep_start, shift_minutes), 0.0)

                # Reset tool and effective age after repair
                target_cnc.wear_index = 0.0
                target_cnc.thermal_index = 0.18
                target_cnc.effective_age = 0.0
                target_cnc.last_state_update = env.now
                target_cnc.is_under_repair = False

                # Finish remaining cut
                rem_start = env.now
                target_cnc.current_job_end_time = rem_start + partial_t
                yield env.timeout(partial_t)
                log_gantt(target_cnc.name, "Processing", rem_start, env.now, f"Job {j_id} (Completed)")
                target_cnc.busy_time += max(min(env.now, shift_minutes) - min(rem_start, shift_minutes), 0.0)
            else:
                # Normal uninterrupted cut
                yield env.timeout(actual_cut_t)
                log_gantt(target_cnc.name, "Processing", cut_start, env.now, f"Job {j_id}")
                target_cnc.busy_time += max(min(env.now, shift_minutes) - min(cut_start, shift_minutes), 0.0)
                target_cnc.effective_age = next_eff_age

            # Update internal condition variables post-cut
            target_cnc.wear_index = min(target_cnc.wear_index + w_inc, 1.15)
            target_cnc.thermal_index = min(target_cnc.thermal_index + th_inc, 1.15)
            target_cnc.last_state_update = env.now
            target_cnc.parts_completed += 1
            # Spindle power model: ~4.5 kW base + 1.8 kW * wear + 1.2 kW * thermal during cut
            power_kw = 4.5 + 1.8 * target_cnc.wear_index + 1.2 * target_cnc.thermal_index
            target_cnc.energy_kwh += power_kw * (actual_cut_t / 60.0)

            cnc_exit_wear = target_cnc.wear_index
            cnc_exit_thermal = target_cnc.thermal_index

        if env.now >= shift_minutes:
            active_wip -= 1
            return

        # ----------------------------------------------------------------------
        # STAGE 3: AUTOMATED OPTICAL INSPECTION (AOI) STATION
        # ----------------------------------------------------------------------
        with aoi_station.request() as req_aoi:
            yield req_aoi
            if env.now >= shift_minutes:
                active_wip -= 1
                return
            aoi_start = env.now
            yield env.timeout(aoi_t)
            log_gantt("AOI Inspection", "Processing", aoi_start, env.now, f"Job {j_id}")

        # Defect probability increases with tool wear and thermal distortion at time of cut
        defect_prob = 0.015 + 0.14 * max(cnc_exit_wear - 0.65, 0.0) + 0.10 * max(cnc_exit_thermal - 0.70, 0.0)
        is_defective = q_roll < defect_prob

        if is_defective:
            scrapped_parts += 1
            active_wip -= 1
            wip_trace.append((env.now, active_wip))
            job_records.append({
                "job_id": j_id,
                "assigned_cnc": target_cnc.name,
                "arrival_time": arr_t,
                "completion_time": env.now,
                "lead_time": env.now - arr_t,
                "cnc_wait": wait_cnc,
                "status": "Scrapped at AOI"
            })
            return

        # ----------------------------------------------------------------------
        # STAGE 4 & 5: COBOT PALLETIZING STATION & FINISHED-GOODS OUTLET
        # ----------------------------------------------------------------------
        with cobot_station.request() as req_cobot:
            yield req_cobot
            if env.now >= shift_minutes:
                active_wip -= 1
                return
            cobot_start = env.now
            yield env.timeout(cobot_t)
            log_gantt("Cobot Palletizer", "Processing", cobot_start, env.now, f"Job {j_id}")

        if env.now <= shift_minutes:
            good_parts_outlet += 1
            active_wip -= 1
            wip_trace.append((env.now, active_wip))
            job_records.append({
                "job_id": j_id,
                "assigned_cnc": target_cnc.name,
                "arrival_time": arr_t,
                "completion_time": env.now,
                "lead_time": env.now - arr_t,
                "cnc_wait": wait_cnc,
                "status": "Good (Finished Outlet)"
            })

    def telemetry_sampler():
        """Samples internal condition states every 2 minutes across the 480-minute shift."""
        while env.now < shift_minutes:
            for m in cnc_machines:
                m.update_Cooling()
                telemetry_logs.append({
                    "Time": env.now,
                    "Machine": m.name,
                    "Wear_Index": m.wear_index,
                    "Thermal_Index": m.thermal_index,
                    "Queue_Length": len(m.resource.queue),
                    "Under_Repair": int(m.is_under_repair)
                })
            yield env.timeout(2.0)

    # Register processes
    for _, row in jobs_df.iterrows():
        env.process(process_job(row))
    env.process(telemetry_sampler())

    # Run 8-hour shift (480 minutes)
    env.run(until=shift_minutes)

    # --------------------------------------------------------------------------
    # KPI & OEE CALCULATIONS (PROPERLY DEFINED AVAILABILITY x PERFORMANCE x QUALITY)
    # --------------------------------------------------------------------------
    total_planned_cnc_time = 2.0 * shift_minutes  # 2 parallel CNCs * 480 min = 960 machine-mins
    total_downtime = sum(m.downtime for m in cnc_machines)
    total_busy_time = sum(m.busy_time for m in cnc_machines)
    operating_time = max(total_planned_cnc_time - total_downtime, 1.0)

    # 1. Availability = Operating Time / Planned Production Time
    availability = np.clip(operating_time / total_planned_cnc_time, 0.0, 1.0)

    # 2. Performance = (Total Parts Produced * Ideal Cycle Time) / Operating Time
    total_cnc_parts = sum(m.parts_completed for m in cnc_machines)
    ideal_cycle_time = 3.6  # Nominal ideal cycle time [min]
    performance = np.clip((total_cnc_parts * ideal_cycle_time) / operating_time, 0.0, 1.0)

    # 3. Quality = Good Parts at Outlet / Total Inspected Parts
    total_inspected = good_parts_outlet + scrapped_parts
    quality = (good_parts_outlet / total_inspected) if total_inspected > 0 else 1.0

    oee = availability * performance * quality

    # Time-weighted Average WIP over full shift & steady-state window [60, 420] min
    wip_df = pd.DataFrame(wip_trace, columns=["t", "wip"])
    if not wip_df.empty:
        wip_df["dt"] = wip_df["t"].diff().shift(-1).fillna(shift_minutes - wip_df["t"].iloc[-1])
        avg_wip = float(np.sum(wip_df["wip"] * wip_df["dt"]) / shift_minutes)

        # Steady-state window [60, 420] min for Little's Law check
        ss_df = wip_df[(wip_df["t"] >= 60.0) & (wip_df["t"] <= 420.0)].copy()
        if not ss_df.empty:
            ss_df["dt"] = ss_df["t"].diff().shift(-1).fillna(420.0 - ss_df["t"].iloc[-1])
            ss_wip_L = float(np.sum(ss_df["wip"] * ss_df["dt"]) / 360.0)
        else:
            ss_wip_L = avg_wip
    else:
        avg_wip, ss_wip_L = 0.0, 0.0

    jobs_out_df = pd.DataFrame(job_records)
    if not jobs_out_df.empty:
        avg_lead_time = float(jobs_out_df["lead_time"].mean())
        ss_jobs = jobs_out_df[(jobs_out_df["completion_time"] >= 60.0) & (jobs_out_df["completion_time"] <= 420.0)]
        ss_W = float(ss_jobs["lead_time"].mean()) if not ss_jobs.empty else avg_lead_time
        ss_lambda = len(ss_jobs) / 360.0
    else:
        avg_lead_time, ss_W, ss_lambda = 0.0, 0.0, 0.0

    # Idle standby power (0.8 kW per CNC when not cutting) + AOI/Cobot station energy
    standby_kwh = 0.8 * ((total_planned_cnc_time - total_busy_time) / 60.0)
    aux_kwh = 0.12 * total_inspected
    total_energy_kwh = sum(m.energy_kwh for m in cnc_machines) + standby_kwh + aux_kwh
    energy_per_good_part = total_energy_kwh / max(good_parts_outlet, 1)

    kpis = {
        "throughput_good": good_parts_outlet,
        "scrapped_parts": scrapped_parts,
        "oee_pct": oee * 100.0,
        "availability_pct": availability * 100.0,
        "performance_pct": performance * 100.0,
        "quality_pct": quality * 100.0,
        "avg_wip": avg_wip,
        "avg_lead_time": avg_lead_time,
        "downtime_min": total_downtime,
        "unplanned_breakdowns": sum(m.unplanned_breakdowns for m in cnc_machines),
        "preventive_maint_count": sum(m.preventive_replacements for m in cnc_machines),
        "energy_per_good_part": energy_per_good_part,
        "ss_L_observed": ss_wip_L,
        "ss_lambda": ss_lambda,
        "ss_W": ss_W,
        "ss_lambda_W": ss_lambda * ss_W
    }

    return {
        "kpis": kpis,
        "gantt_df": pd.DataFrame(gantt_logs),
        "telemetry_df": pd.DataFrame(telemetry_logs),
        "jobs_out_df": jobs_out_df
    }


# ==============================================================================
# 3. PAIRED MONTE CARLO REPLICATION RUNNER (N = 20-30 REPLICATIONS)
# ==============================================================================
@st.cache_data(show_spinner="Running Paired Common Random Number (CRN) Replications...")
def run_monte_carlo_replications(
    n_reps: int,
    num_jobs: int,
    shift_minutes: float,
    weibull_beta: float,
    weibull_eta: float,
    alpha_wear: float,
    beta_thermal: float,
    pm_wear_threshold: float,
    base_seed: int
):
    rep_rows = []
    rep1_details = None

    for r in range(n_reps):
        seed_r = base_seed + r * 101
        jobs_df, machine_crn = generate_crn_inputs(
            num_jobs=num_jobs,
            shift_minutes=shift_minutes,
            weibull_beta=weibull_beta,
            weibull_eta=weibull_eta,
            seed=seed_r
        )

        res_fixed = run_cell_simulation(
            strategy="fixed",
            jobs_df=jobs_df,
            machine_crn=machine_crn,
            shift_minutes=shift_minutes,
            alpha_wear=alpha_wear,
            beta_thermal=beta_thermal,
            pm_wear_threshold=pm_wear_threshold
        )
        res_smart = run_cell_simulation(
            strategy="smart",
            jobs_df=jobs_df,
            machine_crn=machine_crn,
            shift_minutes=shift_minutes,
            alpha_wear=alpha_wear,
            beta_thermal=beta_thermal,
            pm_wear_threshold=pm_wear_threshold
        )

        if r == 0:
            rep1_details = {
                "jobs_crn_df": jobs_df,
                "fixed": res_fixed,
                "smart": res_smart
            }

        for strat_label, res_obj in [("Fixed Routing (Exp A)", res_fixed), ("Smart Dispatching (Exp B)", res_smart)]:
            row = {"Replication": r + 1, "Strategy": strat_label}
            row.update(res_obj["kpis"])
            rep_rows.append(row)

    mc_df = pd.DataFrame(rep_rows)
    return mc_df, rep1_details


# ==============================================================================
# MAIN STREAMLIT DASHBOARD UI
# ==============================================================================
def main():
    st.sidebar.title("Simulation Parameters")
    st.sidebar.caption("5-Stage Manufacturing Cell (SimPy DES)")

    st.sidebar.markdown("### 1. Shift & Monte Carlo Setup")
    num_jobs = st.sidebar.slider("Jobs per 8-Hour Shift", min_value=140, max_value=240, value=200, step=10)
    n_reps = st.sidebar.slider("Paired Replications (N)", min_value=5, max_value=35, value=25, step=5)
    base_seed = st.sidebar.number_input("Base Random Seed (CRN)", min_value=1, max_value=99999, value=2026)

    st.sidebar.markdown("### 2. Dispatching Weights (Exp B)")
    st.sidebar.latex(r"S_i = \hat{T}_i + \alpha W_i + \beta \Theta_i")
    alpha_wear = st.sidebar.slider("Wear Weight (α)", min_value=0.0, max_value=30.0, value=14.0, step=2.0)
    beta_thermal = st.sidebar.slider("Thermal Weight (β)", min_value=0.0, max_value=30.0, value=10.0, step=2.0)
    pm_wear_threshold = st.sidebar.slider("Preventive Tool Change Limit (W_max)", min_value=0.75, max_value=0.98, value=0.88, step=0.02)

    st.sidebar.markdown("### 3. Weibull Reliability Model")
    weibull_beta = st.sidebar.slider("Weibull Shape (β_shape > 1 = Wear-Out)", min_value=1.2, max_value=4.0, value=2.4, step=0.2)
    weibull_eta = st.sidebar.slider("Weibull Scale / Characteristic Life η [min]", min_value=90.0, max_value=220.0, value=135.0, step=15.0)

    mc_df, rep1 = run_monte_carlo_replications(
        n_reps=n_reps,
        num_jobs=num_jobs,
        shift_minutes=480.0,
        weibull_beta=weibull_beta,
        weibull_eta=weibull_eta,
        alpha_wear=alpha_wear,
        beta_thermal=beta_thermal,
        pm_wear_threshold=pm_wear_threshold,
        base_seed=int(base_seed)
    )

    # Header
    st.title("Discrete-Event Shop Floor Simulation: Fixed vs. Condition-Aware Dispatching")
    st.markdown(
        "**Cell Architecture:** Raw Material Buffer $\\rightarrow$ 2 Parallel CNC Turning Centres "
        "$\\rightarrow$ Automated Optical Inspection (AOI) $\\rightarrow$ Cobot Palletizer $\\rightarrow$ Finished-Goods Outlet "
        f"| **Shift:** 480 min (8 Hours) | **Paired CRN Replications:** $N = {n_reps}$"
    )

    # Compute Summary Means across Replications
    df_fixed = mc_df[mc_df["Strategy"] == "Fixed Routing (Exp A)"]
    df_smart = mc_df[mc_df["Strategy"] == "Smart Dispatching (Exp B)"]

    m_tp_f, m_tp_s = df_fixed["throughput_good"].mean(), df_smart["throughput_good"].mean()
    m_oee_f, m_oee_s = df_fixed["oee_pct"].mean(), df_smart["oee_pct"].mean()
    m_wip_f, m_wip_s = df_fixed["avg_wip"].mean(), df_smart["avg_wip"].mean()
    m_lt_f, m_lt_s = df_fixed["avg_lead_time"].mean(), df_smart["avg_lead_time"].mean()
    m_dt_f, m_dt_s = df_fixed["downtime_min"].mean(), df_smart["downtime_min"].mean()
    m_en_f, m_en_s = df_fixed["energy_per_good_part"].mean(), df_smart["energy_per_good_part"].mean()

    # Top 6 KPI Comparison Cards (Mean across N replications)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Throughput (Good Parts)", f"{m_tp_s:.1f}", delta=f"{m_tp_s - m_tp_f:+.1f} vs Fixed")
    c2.metric("Cell OEE (%)", f"{m_oee_s:.1f}%", delta=f"{m_oee_s - m_oee_f:+.1f}% vs Fixed")
    c3.metric("Average WIP (Parts)", f"{m_wip_s:.2f}", delta=f"{m_wip_s - m_wip_f:+.2f} vs Fixed", delta_color="inverse")
    c4.metric("Average Lead Time", f"{m_lt_s:.1f} min", delta=f"{m_lt_s - m_lt_f:+.1f} min", delta_color="inverse")
    c5.metric("Machine Downtime", f"{m_dt_s:.1f} min", delta=f"{m_dt_s - m_dt_f:+.1f} min", delta_color="inverse")
    c6.metric("Energy / Good Part", f"{m_en_s:.3f} kWh", delta=f"{m_en_s - m_en_f:+.3f} kWh", delta_color="inverse")

    # 5 Structured Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Experimental KPI Comparison (N Replications)",
        "2. Machine Operation Gantt Chart",
        "3. CNC Condition Telemetry (Wear, Thermal & Queue)",
        "4. OEE Breakdown & Little's Law Verification",
        "5. Cell Architecture & FlexSim 2026 Export"
    ])

    # ==========================================================================
    # TAB 1: EXPERIMENTAL KPI COMPARISON TABLE & PAIRED DISTRIBUTIONS
    # ==========================================================================
    with tab1:
        st.subheader(f"Experimental KPI Comparison Across N = {n_reps} Paired CRN Replications")
        st.markdown(
            "Both routing strategies are evaluated on identical pre-generated Common Random Number (CRN) job arrival streams, "
            "processing requirements, and Weibull life thresholds. Paired two-sample $t$-tests quantify statistical significance."
        )

        kpi_specs = [
            ("Throughput (Good Parts / Shift)", "throughput_good", "{:.2f}"),
            ("Overall Equipment Effectiveness — OEE (%)", "oee_pct", "{:.2f}%"),
            ("Availability Component (%)", "availability_pct", "{:.2f}%"),
            ("Performance Component (%)", "performance_pct", "{:.2f}%"),
            ("Quality Component (%)", "quality_pct", "{:.2f}%"),
            ("Average Work-In-Process — WIP (Parts)", "avg_wip", "{:.2f}"),
            ("Average Job Lead Time (min)", "avg_lead_time", "{:.2f}"),
            ("Total CNC Downtime (min / Shift)", "downtime_min", "{:.2f}"),
            ("Unplanned Weibull Breakdowns (Count)", "unplanned_breakdowns", "{:.2f}"),
            ("Energy Consumption per Good Part (kWh)", "energy_per_good_part", "{:.3f}")
        ]

        comp_rows = []
        for label, col, fmt in kpi_specs:
            a_vals = df_fixed[col].to_numpy()
            b_vals = df_smart[col].to_numpy()
            mean_a, sd_a = np.mean(a_vals), np.std(a_vals, ddof=1)
            mean_b, sd_b = np.mean(b_vals), np.std(b_vals, ddof=1)
            diff = mean_b - mean_a
            pct_change = (diff / mean_a * 100.0) if abs(mean_a) > 1e-9 else 0.0
            _, p_val = stats.ttest_rel(b_vals, a_vals)

            comp_rows.append({
                "KPI Metric": label,
                "Exp A: Fixed Routing (Mean ± SD)": f"{fmt.format(mean_a)} ± {sd_a:.2f}",
                "Exp B: Smart Dispatching (Mean ± SD)": f"{fmt.format(mean_b)} ± {sd_b:.2f}",
                "Mean Difference (B − A)": f"{diff:+.2f} ({pct_change:+.1f}%)",
                "Paired t-test p-value": f"{p_val:.4e}" if p_val < 0.001 else f"{p_val:.4f}"
            })

        st.dataframe(pd.DataFrame(comp_rows), hide_index=True)

        # Boxplots comparing distributions across replications
        fig_box = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                "OEE (%) Across Replications",
                "Average Lead Time [min] Across Replications",
                "Total CNC Downtime [min] Across Replications"
            )
        )
        for strat_name, color_hex in [("Fixed Routing (Exp A)", "#DC2626"), ("Smart Dispatching (Exp B)", "#059669")]:
            sub = mc_df[mc_df["Strategy"] == strat_name]
            fig_box.add_trace(go.Box(y=sub["oee_pct"], name=strat_name, marker_color=color_hex, boxmean=True), row=1, col=1)
            fig_box.add_trace(go.Box(y=sub["avg_lead_time"], name=strat_name, marker_color=color_hex, boxmean=True, showlegend=False), row=1, col=2)
            fig_box.add_trace(go.Box(y=sub["downtime_min"], name=strat_name, marker_color=color_hex, boxmean=True, showlegend=False), row=1, col=3)

        fig_box.update_yaxes(title_text="OEE [%]", row=1, col=1)
        fig_box.update_yaxes(title_text="Lead Time [min]", row=1, col=2)
        fig_box.update_yaxes(title_text="Downtime [min]", row=1, col=3)
        fig_box.update_layout(height=400, margin=dict(l=40, r=20, t=40, b=30))
        st.plotly_chart(fig_box)

    # ==========================================================================
    # TAB 2: MACHINE OPERATION GANTT CHART
    # ==========================================================================
    with tab2:
        st.subheader("Machine Operation Timeline (Gantt Chart — Replication 1)")
        st.markdown(
            "Visualizes processing blocks, unplanned Weibull breakdowns, and preventive insert changes across the 480-minute shift. "
            "Notice how **Fixed Routing (Exp A)** suffers long unplanned repair blocks while jobs stay trapped in the broken machine's queue."
        )

        gantt_window = st.slider(
            "Select Shift Time Window to Inspect [minutes]",
            min_value=0, max_value=480, value=(0, 240), step=30
        )

        color_map = {
            "Processing": "#2563EB",
            "Unplanned Breakdown": "#DC2626",
            "Preventive Tool Change": "#F59E0B"
        }
        station_order = ["Cobot Palletizer", "AOI Inspection", "CNC 2", "CNC 1"]

        for title_label, sim_key in [
            ("Experiment A: Conventional Fixed Alternating Routing", "fixed"),
            ("Experiment B: Condition-Aware Dynamic Dispatching", "smart")
        ]:
            gdf = rep1[sim_key]["gantt_df"]
            gdf_win = gdf[(gdf["Finish"] >= gantt_window[0]) & (gdf["Start"] <= gantt_window[1])].copy()

            fig_g = go.Figure()
            for state_label, c_hex in color_map.items():
                state_df = gdf_win[gdf_win["State"] == state_label]
                if state_df.empty:
                    continue
                fig_g.add_trace(go.Bar(
                    base=state_df["Start"],
                    x=state_df["Duration"],
                    y=state_df["Station"],
                    orientation="h",
                    name=state_label,
                    marker_color=c_hex,
                    hovertext=state_df["Details"],
                    hoverinfo="text+x+y"
                ))

            fig_g.update_layout(
                title=title_label,
                barmode="overlay",
                height=290,
                xaxis=dict(title="Shift Time [minutes]", range=[gantt_window[0], gantt_window[1]]),
                yaxis=dict(title="Workstation", categoryorder="array", categoryarray=station_order),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=20, t=50, b=35)
            )
            st.plotly_chart(fig_g)

    # ==========================================================================
    # TAB 3: INTERNAL CONDITION TELEMETRY (WEAR, THERMAL & QUEUE)
    # ==========================================================================
    with tab3:
        st.subheader("Simulated Machine Condition Variables (Replication 1)")
        st.markdown(
            "Tracks the three internal state variables updated during simulation: "
            "**Normalized Tool Wear Index ($W_i$)**, **Spindle Thermal Load ($\Theta_i$)**, and **Machine Queue Length ($Q_i$)**."
        )

        t_fixed = rep1["fixed"]["telemetry_df"]
        t_smart = rep1["smart"]["telemetry_df"]

        fig_tel = make_subplots(
            rows=3, cols=2,
            shared_xaxes=True,
            vertical_spacing=0.10,
            subplot_titles=(
                "Exp A (Fixed): Tool Wear Index W_i",
                "Exp B (Smart): Tool Wear Index W_i",
                "Exp A (Fixed): Spindle Thermal Load Θ_i",
                "Exp B (Smart): Spindle Thermal Load Θ_i",
                "Exp A (Fixed): CNC Queue Length [Jobs]",
                "Exp B (Smart): CNC Queue Length [Jobs]"
            )
        )

        for col_idx, t_df in [(1, t_fixed), (2, t_smart)]:
            for m_name, c_hex in [("CNC 1", "#2563EB"), ("CNC 2", "#D97706")]:
                m_sub = t_df[t_df["Machine"] == m_name]
                show_leg = (col_idx == 1)
                fig_tel.add_trace(go.Scatter(
                    x=m_sub["Time"], y=m_sub["Wear_Index"],
                    name=m_name, line=dict(color=c_hex, width=1.8), showlegend=show_leg
                ), row=1, col=col_idx)

                fig_tel.add_trace(go.Scatter(
                    x=m_sub["Time"], y=m_sub["Thermal_Index"],
                    name=m_name, line=dict(color=c_hex, width=1.8), showlegend=False
                ), row=2, col=col_idx)

                fig_tel.add_trace(go.Scatter(
                    x=m_sub["Time"], y=m_sub["Queue_Length"],
                    name=m_name, line=dict(color=c_hex, width=1.8), fill="tozeroy", showlegend=False
                ), row=3, col=col_idx)

        fig_tel.update_xaxes(title_text="Shift Time [minutes]", row=3, col=1)
        fig_tel.update_xaxes(title_text="Shift Time [minutes]", row=3, col=2)
        fig_tel.update_yaxes(title_text="Wear W_i", row=1, col=1)
        fig_tel.update_yaxes(title_text="Thermal Θ_i", row=2, col=1)
        fig_tel.update_yaxes(title_text="Queue [Jobs]", row=3, col=1)

        fig_tel.update_layout(height=620, margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig_tel)

    # ==========================================================================
    # TAB 4: OEE DECOMPOSITION & LITTLE'S LAW VERIFICATION
    # ==========================================================================
    with tab4:
        col_oee, col_little = st.columns([1.15, 1.0])

        with col_oee:
            st.subheader("OEE Component Decomposition")
            st.latex(r"\text{OEE} = \text{Availability} \times \text{Performance} \times \text{Quality}")

            oee_metrics = ["availability_pct", "performance_pct", "quality_pct", "oee_pct"]
            oee_labels = ["Availability (%)", "Performance (%)", "Quality (%)", "Total OEE (%)"]

            fig_oee = go.Figure()
            fig_oee.add_trace(go.Bar(
                x=oee_labels,
                y=[df_fixed[m].mean() for m in oee_metrics],
                name="Exp A: Fixed Routing",
                marker_color="#64748B",
                text=[f"{df_fixed[m].mean():.1f}%" for m in oee_metrics],
                textposition="auto"
            ))
            fig_oee.add_trace(go.Bar(
                x=oee_labels,
                y=[df_smart[m].mean() for m in oee_metrics],
                name="Exp B: Smart Dispatching",
                marker_color="#059669",
                text=[f"{df_smart[m].mean():.1f}%" for m in oee_metrics],
                textposition="auto"
            ))
            fig_oee.update_layout(
                barmode="group",
                height=360,
                xaxis_title="OEE Factor",
                yaxis_title="Percentage [%]",
                yaxis_range=[0, 108],
                margin=dict(l=40, r=20, t=20, b=40)
            )
            st.plotly_chart(fig_oee)

        with col_little:
            st.subheader("Little's Law Steady-State Consistency Check")
            st.latex(r"L = \lambda \cdot W \quad \text{(Evaluated on Steady-State Window } t \in [60, 420]\text{ min)}")
            st.markdown(
                "A finite 480-minute shift starts empty ($WIP=0$) and may have unfinished jobs at $t=480\text{ min}$. "
                "Evaluating over the interior steady-state window ($60\text{--}420\text{ min}$) verifies queueing consistency:"
            )

            little_rows = []
            for strat_label, sub_df in [("Exp A: Fixed Routing", df_fixed), ("Exp B: Smart Dispatching", df_smart)]:
                L_obs = sub_df["ss_L_observed"].mean()
                lam = sub_df["ss_lambda"].mean()
                W_obs = sub_df["ss_W"].mean()
                L_calc = sub_df["ss_lambda_W"].mean()
                discrepancy_pct = abs(L_obs - L_calc) / max(L_obs, 1e-6) * 100.0
                little_rows.append({
                    "Strategy": strat_label,
                    "Throughput Rate λ [jobs/min]": f"{lam:.4f}",
                    "Mean Time in System W [min]": f"{W_obs:.2f}",
                    "Theoretical λ·W [parts]": f"{L_calc:.2f}",
                    "Time-Averaged Observed L [parts]": f"{L_obs:.2f}",
                    "Discrepancy (%)": f"{discrepancy_pct:.2f}%"
                })
            st.dataframe(pd.DataFrame(little_rows), hide_index=True)

    # ==========================================================================
    # TAB 5: CELL ARCHITECTURE & FLEXSIM 2026 EXPORT
    # ==========================================================================
    with tab5:
        st.subheader("Cell Modelling Specification & FlexSim 2026 Schedule Export")

        specs = [
            (
                "1. Five-Stage Manufacturing Cell Topology",
                "Raw Material Buffer -> Two Parallel CNC Turning Centres (CNC 1 & CNC 2) -> Automated Optical Inspection (AOI) Station -> Cobot Palletizing Station -> Finished-Goods Outlet."
            ),
            (
                "2. Coupled Degradation & Weibull Reliability Model",
                "Rather than claiming that condition tracking prevents independent failures, each CNC's effective operating age advances with wear and heat: Δu_i = Δt · (1 + 1.8 W_i² + 1.5 Θ_i²). Unplanned breakdowns occur when u_i crosses a pre-sampled Weibull threshold L = η(-ln U)^(1/β)."
            ),
            (
                "3. Condition-Aware Dispatching Function (Experiment B)",
                "Whenever a job arrives, the supervisory controller evaluates S_i = T_hat_i + α W_i + β Θ_i for i in {1, 2}, selecting the CNC with the lower score and triggering a short preventive insert replacement when W_i reaches the threshold."
            )
        ]
        for s_t, s_b in specs:
            st.markdown(
                f'<div class="spec-card"><div class="spec-title">{s_t}</div><div class="spec-body">{s_b}</div></div>',
                unsafe_allow_html=True
            )

        # Prepare FlexSim-compatible CSV table from Replication 1
        crn_df = rep1["jobs_crn_df"].copy()
        smart_out = rep1["smart"]["jobs_out_df"][["job_id", "assigned_cnc", "lead_time", "status"]]
        flexsim_export_df = pd.merge(crn_df, smart_out, on="job_id", how="left")

        st.markdown("#### Download Pre-Generated CRN Job Schedule for FlexSim 2026 Import")
        csv_bytes = flexsim_export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download FlexSim 2026 Arrival & Dispatch Table (flexsim_2026_schedule.csv)",
            data=csv_bytes,
            file_name="flexsim_2026_schedule.csv",
            mime="text/csv"
        )
        st.dataframe(flexsim_export_df.head(25), hide_index=True)


if __name__ == "__main__":
    main()