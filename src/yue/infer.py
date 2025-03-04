import os
import sys
from datetime import datetime
from common import parser


def check_exit(status: int):
    if status != 0:
        sys.exit(status)


if __name__ == "__main__":
    parser.parse_args()  # make --help work
    dirname = os.path.dirname(os.path.abspath(__file__))
    
    generation_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    print("Starting stage 1...")
    check_exit(os.system(f'python3 {os.path.join(dirname, "infer_stage1.py")} {" ".join(sys.argv[1:])} --generation_timestamp {generation_timestamp}'))
    print("Starting stage 2...")
    check_exit(os.system(f'python3 {os.path.join(dirname, "infer_stage2.py")} {" ".join(sys.argv[1:])} --generation_timestamp {generation_timestamp}'))
    print("Starting postprocessing...")
    check_exit(os.system(f'python3 {os.path.join(dirname, "infer_postprocess.py")} {" ".join(sys.argv[1:])} --generation_timestamp {generation_timestamp}'))
