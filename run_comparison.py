import subprocess
import os
import sys

# --- CONFIGURATION ---
# Chemin vers le dossier racine du repo "Vanilla"
VANILLA_REPO_PATH = "/Users/aedelon/Workspace/depth-anything-3-upstream"


# ---------------------

def run_in_env(python_path_prepend, label):
    print(f"\n{'-' * 10} RUNNING: {label} {'-' * 10}")
    print(f"Using source: {python_path_prepend}")

    env = os.environ.copy()
    # On force le PYTHONPATH pour charger la version spécifique de la lib
    if python_path_prepend:
        current_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{python_path_prepend}:{current_pp}"

    cmd = [sys.executable, "benchmark_payload.py"]

    try:
        # Exécution du payload dans un processus séparé
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)

        # Affichage des logs du sous-processus pour débogage
        print(result.stdout)
        if result.returncode != 0:
            print("❌ Error during execution:")
            print(result.stderr)
            return None

        # Parsing du résultat
        for line in result.stdout.splitlines():
            if "RESULT_TIME=" in line:
                return float(line.split("=")[1])
        return None

    except Exception as e:
        print(f"❌ Execution failed: {e}")
        return None


def main():
    if not os.path.exists(VANILLA_REPO_PATH):
        print(f"❌ Vanilla path not found: {VANILLA_REPO_PATH}")
        return

    cwd = os.getcwd()

    # 1. Benchmark OPTIMIZED (Dossier actuel)
    # On pointe vers ./src pour charger 'depth_anything_3' d'ici
    t_opt = run_in_env(os.path.join(cwd, "src"), "🟢 OPTIMIZED (Dynamic Batching)")

    # 2. Benchmark VANILLA (Dossier upstream)
    # On pointe vers upstream/src
    t_vanilla = run_in_env(os.path.join(VANILLA_REPO_PATH, "src"), "🔴 VANILLA (Upstream)")

    # 3. Résultats
    print(f"\n{'=' * 30}")
    print("       BENCHMARK RESULTS       ")
    print(f"{'=' * 30}")

    if t_opt and t_vanilla:
        print(f"Vanilla Time:   {t_vanilla:.4f} s")
        print(f"Optimized Time: {t_opt:.4f} s")
        print(f"{'-' * 30}")

        diff = t_vanilla - t_opt
        if diff > 0:
            speedup = t_vanilla / t_opt
            percent = (diff / t_vanilla) * 100
            print(f"✅ IMPROVEMENT: {diff:.4f} s faster")
            print(f"🚀 SPEEDUP:     {speedup:.2f}x ({percent:.1f}% reduction)")
        else:
            print(f"⚠️ REGRESSION:  {abs(diff):.4f} s slower")
    else:
        print("❌ Could not compute comparison (one run failed).")


if __name__ == "__main__":
    main()