import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from graph import run_query

q = "How actually today's chatbots use transformer arch to give this much optimal responses?"
print(f"Testing Query: {q}\n")

res = run_query(q)

print("=" * 60)
print("FINAL OUTPUT:")
print("=" * 60)
print(res["final_output"])
print("=" * 60)
print("VERIFICATION RESULT:")
print(res["verification_result"])
print("=" * 60)
