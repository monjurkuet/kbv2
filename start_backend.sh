#!/bin/bash
cd /home/muham/development/kbv2
export PYTHONPATH=./src
python3 -c "
import sys
sys.path.insert(0, 'src')
from knowledge_base.main import app
import uvicorn
uvicorn.run(app, host='0.0.0.0', port=8765, reload=False)
" > backend.log 2>&1
echo "Backend started with PID: $!"
