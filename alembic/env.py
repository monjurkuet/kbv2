from logging.config import fileConfig

from sqlalchemy import engine_from_config, create_engine
from sqlalchemy import pool

from alembic import context

import sys
import os

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded .env file for Alembic")
except ImportError:
    print("dotenv not available, skipping .env load")

# Explicitly read DATABASE_URL from environment
database_url = os.getenv("DATABASE_URL")
if database_url:
    print(f"Using DATABASE_URL: {database_url.split('@')[1] if '@' in database_url else 'localhost'}")
    # Set it in the alembic config
    config = context.config
    if config:
        config.set_main_option("sqlalchemy.url", database_url)
        print("Set sqlalchemy.url from DATABASE_URL")
else:
    print("WARNING: DATABASE_URL not set in environment")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.knowledge_base.persistence.v1.schema import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Now read the URL again (might have been set above)
url = config.get_main_option("sqlalchemy.url")
print(f"Final sqlalchemy.url: {url.split('@')[1] if '@' in url else url}")

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
