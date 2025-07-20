import sys
import os

def open_heatmap_dir():
    import os
    import subprocess
    # snapshots 폴더 경로
    #base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #snapshot_dir = os.path.join(base_dir, "snapshots")
    
    snapshot_dir = "C:/Users/orgin/XAI-study/heatmap_tool/app/snapshots"
    os.makedirs(snapshot_dir, exist_ok=True)
    if sys.platform.startswith('win'):
        os.startfile(snapshot_dir)
    elif sys.platform.startswith('darwin'):
        subprocess.Popen(['open', snapshot_dir])
    else:
        subprocess.Popen(['xdg-open', snapshot_dir])