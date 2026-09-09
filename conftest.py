import pytest


# Suppress pytest-asyncio deprecation warning about fixture loop scope
# by configuring the default loop scope explicitly.
# This file is picked up automatically by pytest at the root level.
