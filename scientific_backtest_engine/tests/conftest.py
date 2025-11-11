import sys
from pathlib import Path
# 將倉庫根目錄加入 Python 路徑，確保能以包形式導入
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
