import sys
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("verb", help="installed/missing")
parser.add_argument(
    "packages", help="list of items to be there/not be there", nargs="+"
)

if __name__ == "__main__":
    args = parser.parse_args()
    installed = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode(
        "utf-8"
    )
    for pkg in args.packages:
        if args.verb == "missing":
            if pkg in installed:
                raise ValueError(f"Expected {pkg} to not be installed.")
        if args.verb == "installed":
            if pkg not in installed:
                raise ValueError(f"Expected {pkg} to be installed.")
