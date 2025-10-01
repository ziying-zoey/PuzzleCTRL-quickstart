import json, glob
import matplotlib.pyplot as plt

def load_summaries():
    data=[]
    for fp in glob.glob("logs/summary_*.json"):
        with open(fp,'r') as f:
            j=json.load(f)
        a=j["aggregate"]; m=j["config"]["method"]
        data.append({"method":m,"tool":a["avg_tool_calls"],"time":a["avg_time_sec"],"pass":a["pass_at_1_like"]})
    return data

d=load_summaries()
# 图A：pass@1 vs tool_calls
plt.figure()
for m in set(x["method"] for x in d):
    xs=[x["tool"] for x in d if x["method"]==m]
    ys=[x["pass"] for x in d if x["method"]==m]
    plt.plot(xs, ys, marker="o", label=m)
plt.xlabel("avg tool calls"); plt.ylabel("pass@1 (like)"); plt.legend(); plt.title("pass@1 vs tool_calls")
plt.savefig("logs/curve_tool_calls.png", dpi=160)

# 图B：pass@1 vs time
plt.figure()
for m in set(x["method"] for x in d):
    xs=[x["time"] for x in d if x["method"]==m]
    ys=[x["pass"] for x in d if x["method"]==m]
    plt.plot(xs, ys, marker="o", label=m)
plt.xlabel("avg time (s)"); plt.ylabel("pass@1 (like)"); plt.legend(); plt.title("pass@1 vs time")
plt.savefig("logs/curve_time.png", dpi=160)
print("saved plots under logs/")