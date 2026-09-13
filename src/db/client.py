"""
Supabase client — single shared connection.
Import `supabase` from here anywhere you need to read/write the database.
"""

from supabase import create_client, Client
from src.core.config import settings


def get_client() -> Client:
    return create_client(settings.supabase_url, settings.supabase_key)


# Shared instance — import this in most places
supabase: Client = get_client()