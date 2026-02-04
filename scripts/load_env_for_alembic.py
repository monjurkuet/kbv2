#!/usr/bin/env python3
"""Load environment variables for Alembic."""
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Set alembic username from environment if available
if os.getenv('DB_USER'):
    os.environ['PGUSER'] = os.getenv('DB_USER')
if os.getenv('DB_PASSWORD'):
    os.environ['PGPASSWORD'] = os.getenv('DB_PASSWORD')
