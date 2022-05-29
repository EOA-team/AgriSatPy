"""
Creates tables in a PostgreSQL/PostGIS DBMS instance if not available yet
"""

from agrisatpy.metadata.database.db_model import create_tables

if __name__ == '__main__':
    create_tables()
