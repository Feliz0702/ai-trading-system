# AI交易系統 - 第一階段基礎設施搭建

![CI](https://github.com/OWNER/ai-trading-system/actions/workflows/test.yml/badge.svg)
[![codecov](https://codecov.io/gh/OWNER/ai-trading-system/branch/main/graph/badge.svg)](https://codecov.io/gh/OWNER/ai-trading-system)

## 📋 項目概述

本項目是一個基於Docker的AI交易系統基礎設施，包含數據收集、策略引擎和監控面板等服務。

> 提示：上述徽章中的 OWNER 請替換為你的 GitHub 用戶或組織名稱。

## 🖥️ 系統要求

### Windows環境
- Windows 10/11 (64位)
- Docker Desktop for Windows
- PowerShell 5.0 或更高版本

### Linux環境（推薦）
- Ubuntu Server 20.04/22.04 LTS
- Docker 和 Docker Compose

## 🚀 Windows環境安裝步驟

### 步驟1：安裝Docker Desktop

1. 下載 Docker Desktop for Windows：
   - 訪問：https://www.docker.com/products/docker-desktop
   - 下載並安裝Docker Desktop

2. 啟動Docker Desktop並確保它正在運行

3. 驗證安裝：
```powershell
docker --version
docker-compose --version
```

### 步驟2：配置環境變量

1. 編輯 `.env` 文件，設置安全密碼：
```powershell
# 使用記事本或編輯器打開 .env 文件
notepad .env
```

2. 修改以下配置：
```
DB_PASSWORD=secure_password_123
REDIS_PASSWORD=redis_pass_123
```

3. 如需使用實際交易API，請填入相應的API密鑰：
```
BINANCE_API_KEY=your_actual_binance_api_key
BINANCE_SECRET_KEY=your_actual_binance_secret_key
ALPACA_API_KEY=your_actual_alpaca_api_key
ALPACA_SECRET_KEY=your_actual_alpaca_secret_key
```

### 步驟3：構建並啟動服務

```powershell
# 進入項目目錄
cd ai-trading-system

# 構建並啟動所有服務
docker-compose up -d --build

# 檢查服務狀態
docker-compose ps

# 查看日誌
docker-compose logs -f data-collector
```

### 步驟4：測試服務

使用PowerShell測試腳本：
```powershell
.\scripts\test_setup.ps1
```

或使用Python測試腳本（需要安裝Python和依賴）：
```powershell
pip install psycopg2-binary redis python-dotenv
python scripts\test_setup.py
```

## 🐧 Linux環境安裝步驟

### 步驟1：環境準備

```bash
# 系統更新
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget vim htop

# 安裝Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# 安裝Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 驗證安裝
docker --version
docker-compose --version
```

### 步驟2：配置環境變量

```bash
cd ~/ai-trading-system
# 編輯.env文件
nano .env
```

### 步驟3：啟動服務

```bash
# 設置環境變量（如果需要）
echo "DB_PASSWORD=secure_password_123" >> .env
echo "REDIS_PASSWORD=redis_pass_123" >> .env

# 啟動所有服務
docker-compose up -d

# 檢查服務狀態
docker-compose ps

# 查看日誌
docker-compose logs -f data-collector
```

### 步驟4：安裝監控工具（可選）

```bash
# 安裝系統監控
sudo apt install -y htop iotop nethogs

# 安裝進程監控
sudo apt install -y supervisor

# 創建Supervisor配置
sudo tee /etc/supervisor/conf.d/trading-system.conf > /dev/null <<EOF
[program:trading-dashboard]
command=docker-compose -f /home/$(whoami)/ai-trading-system/docker-compose.yml up
directory=/home/$(whoami)/ai-trading-system
autostart=true
autorestart=true
user=$(whoami)
EOF

# 重啟Supervisor
sudo supervisorctl reload
```

## 📁 項目結構

```
ai-trading-system/
├── config/                 # 配置文件
│   ├── brokers/           # 交易商配置
│   ├── strategies/        # 策略配置
│   ├── risk/             # 風險管理配置
│   └── database/         # 數據庫初始化腳本
├── data/                  # 數據目錄
│   ├── market/           # 市場數據
│   ├── portfolio/        # 投資組合數據
│   └── results/          # 結果數據
├── services/              # 服務目錄
│   ├── data-collector/   # 數據收集服務
│   ├── strategy-engine/  # 策略引擎（待實現）
│   └── dashboard/        # Web監控面板（待實現）
├── strategies/            # 交易策略
├── scripts/               # 腳本目錄
├── logs/                  # 日誌目錄
├── reports/               # 報告目錄
├── backtests/             # 回測結果
├── docker-compose.yml     # Docker編排配置
└── .env                   # 環境變量配置
```

## 🔬 科學回測引擎（Scientific Backtest Engine）

科學回測引擎已集成於本倉庫：`scientific_backtest_engine/`

- 安裝依賴
  - 推薦在虛擬環境中安裝

```bash
pip install -r scientific_backtest_engine/requirements.txt
```

- 快速開始（示例）

```python
import pandas as pd
import numpy as np
from scientific_backtest_engine import ScientificBacktestEngine, BacktestConfig

# 生成示例數據
dates = pd.date_range('2020-01-01', periods=400, freq='D')
ret = np.random.normal(0.0005, 0.02, len(dates))
price = 100 * np.cumprod(1 + ret)
data = pd.DataFrame({'close': price, 'open': price, 'high': price*1.01, 'low': price*0.99, 'volume': 1_000}, index=dates)

# 定義簡單策略
def ma_cross(data: pd.DataFrame, params):
    s = params.get('short', 10); l = params.get('long', 30)
    d = data.copy()
    d['ret'] = d['close'].pct_change().fillna(0)
    d['sma_s'] = d['close'].rolling(s).mean()
    d['sma_l'] = d['close'].rolling(l).mean()
    d['sig'] = 0
    d.loc[d['sma_s'] > d['sma_l'], 'sig'] = 1
    d.loc[d['sma_s'] < d['sma_l'], 'sig'] = -1
    d['strategy_returns'] = d['sig'].shift(1).fillna(0) * d['ret']
    return d['strategy_returns'].dropna()

engine = ScientificBacktestEngine(BacktestConfig())
engine.set_strategy(ma_cross).load_data(data)
param_space = {'short': [10, 15], 'long': [30, 40]}
results = engine.run_comprehensive_analysis(param_space)
print(results['final_assessment'])
```

- 內建測試

```bash
pytest scientific_backtest_engine/tests -q
```

### 📊 可視化示例

已提供增強版性能分析可視化（Plotly 交互式）：

- 可視化分析器：`scientific_backtest_engine/analysis/performance_analyzer.py`（`EnhancedPerformanceAnalyzer`）
- 演示腳本：`scientific_backtest_engine/scripts/demo_visualization.py`

運行示例：

```bash
python scientific_backtest_engine/scripts/demo_visualization.py
```

輸出（HTML，位於 `visualization_demo/`）：

- performance_dashboard.html（綜合儀表板）
- rolling_metrics.html（滾動指標）
- stress_distribution.html（壓力測試分佈）

## 🔧 常用命令

### Docker Compose命令

```powershell
# 啟動所有服務
docker-compose up -d

# 停止所有服務
docker-compose down

# 查看服務狀態
docker-compose ps

# 查看日誌
docker-compose logs -f [service_name]

# 重啟服務
docker-compose restart [service_name]

# 重建服務
docker-compose up -d --build [service_name]

# 進入容器
docker-compose exec [service_name] /bin/bash
```

### 數據庫操作

```powershell
# 連接到PostgreSQL
docker-compose exec postgres psql -U trader -d trading

# 執行SQL腳本
docker-compose exec postgres psql -U trader -d trading -f /docker-entrypoint-initdb.d/init.sql
```

### Redis操作

```powershell
# 連接到Redis
docker-compose exec redis redis-cli

# 測試Redis連接
docker-compose exec redis redis-cli ping
```

## ⚠️ 注意事項

1. **安全警告**：`.env` 文件包含敏感信息，請勿提交到版本控制系統
2. **API密鑰**：使用實際交易API時，請確保使用只讀權限的API密鑰進行測試
3. **密碼設置**：生產環境請使用強密碼
4. **資源需求**：確保系統有足夠的內存和磁盤空間運行Docker容器

## 🐛 故障排除

### Docker Desktop未運行
- 確保Docker Desktop已啟動
- 檢查Windows服務中的Docker相關服務是否運行

### 端口衝突
- 如果5432端口被占用，修改 `docker-compose.yml` 中的端口映射
- 如果6379端口被占用，同樣修改Redis的端口映射

### 容器無法啟動
```powershell
# 查看詳細錯誤日誌
docker-compose logs [service_name]

# 檢查容器狀態
docker-compose ps -a
```

## 📝 下一步

完成基礎設施搭建後，您可以：
1. 實現數據收集邏輯
2. 開發交易策略
3. 構建策略引擎
4. 開發Web監控面板

## 📄 許可證

[待定]

