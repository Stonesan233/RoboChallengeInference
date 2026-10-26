# -*- coding: utf-8 -*-
import os

REALSENSE_DEVICE_IDS = None

# ROBOT_TAG / RECORD_DATA_DIR can be overridden by env vars for quick switching.
# Only one pair should be active at a time.
ROBOT_TAG = os.environ.get('MOCK_ROBOT_TAG', 'franka')
RECORD_DATA_DIR = os.environ.get(
    'MOCK_RECORD_DATA_DIR', '../20250919/press_three_buttons/data/episode_FFFFFF'
)

# ROBOT_TAG = 'aloha'
# RECORD_DATA_DIR = '../20250919/pour_fries_into_plate/data/episode_001533'

# ROBOT_TAG = 'arx5'
# RECORD_DATA_DIR = '../20250919/arrange_paper_cups/data/episode_000772'

# ROBOT_TAG = 'ur5'
# RECORD_DATA_DIR = '../20250919/shred_scrap_paper/data/episode_000775'

# ROBOT_TAG = 'franka'
# RECORD_DATA_DIR = '../20250919/press_three_buttons/data/episode_FFFFFF'
