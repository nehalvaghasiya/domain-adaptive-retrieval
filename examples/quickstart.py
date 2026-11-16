"""One-command CPU offline integration check for new contributors."""

from domain_adaptive_retrieval.workflows.smoke import run_smoke

if __name__ == "__main__":
    results = run_smoke("outputs/quickstart")
    for domain, result in results.items():
        print(domain, ", ".join(result["systems"]), "(synthetic only)")
