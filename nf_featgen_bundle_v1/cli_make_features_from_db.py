#!/usr/bin/env python3
from featgen.runner_db import make_features_from_db
import sys
def main():
    cfg = sys.argv[1] if len(sys.argv) > 1 else None
    make_features_from_db(cfg)
if __name__ == "__main__":
    main()
