import pandas as pd

df = pd.read_csv("flexsim_2026_schedule.csv")

flexsim_df = pd.DataFrame({
    "ArrivalTime": df["arrival_time"].round(3),
    "ItemName": "Cylinder",
    "Quantity": 1,
    "job_id": df["job_id"],
    "cnc_port": df["assigned_cnc"].apply(lambda x: 1 if str(x).strip() == "CNC 1" else 2),
    "cnc_proc_time": df["cnc_proc_time"].round(3),
    "aoi_proc_time": df["aoi_proc_time"].round(3),
    "aoi_port": df["status"].apply(lambda x: 2 if "Scrapped" in str(x) else 1),
    "cobot_proc_time": df["cobot_proc_time"].round(3)
})

with pd.ExcelWriter("FlexSim_Ready.xlsx") as writer:
    flexsim_df.to_excel(writer, sheet_name="Sheet1", index=False)

print("Saved FlexSim_Ready.xlsx with Sheet1 and exact FlexSim columns!")