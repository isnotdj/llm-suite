#!/opt/venv/bin/python
"""Print an external benchmark's mapped path for use by its official CLI."""
import argparse
from dataset_mapping import read_mapping, resolve

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('name')
args = parser.parse_args()
print(resolve(args.name, read_mapping()))
