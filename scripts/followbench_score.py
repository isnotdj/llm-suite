"""Run the pinned FollowBench scorer with explicit dataset and I/O exit codes."""
import os
import runpy
import sys
import traceback
from pathlib import Path


if __name__ == '__main__':
    script = Path(sys.argv.pop(1))
    sys.argv[0] = str(script)
    sys.path.insert(0, str(script.parent))
    try:
        runpy.run_path(str(script), run_name='__main__')
    except OSError:
        traceback.print_exc()
        sys.exit(os.EX_IOERR)
    except (ValueError, KeyError, IndexError, AssertionError):
        traceback.print_exc()
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        sys.exit(os.EX_SOFTWARE)
