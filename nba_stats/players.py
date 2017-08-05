#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Players Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""
import io
import logging
import os.path as osp
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

log_format = ('%(asctime)s  %(levelname)8s  -> %(name)s <- '
              '(line: %(lineno)d) %(message)s\n')
date_format = '%m/%d/%Y %I:%M:%S'
logging.basicConfig(format=log_format, datefmt=date_format,
                    level=logging.INFO)

current_dir = osp.dirname(osp.realpath(__file__))
data_dir = osp.realpath(osp.join(current_dir, '..', 'data'))
players_file = osp.join(data_dir, 'Players.csv')
season_file = osp.join(data_dir, 'Seasons_Stats.csv')


class Statistics:
    """
    Methods and attributes related to NBA player statistics.

    ..note:: Original Player and Season Statistics datasets from \
        `kaggle <https://www.kaggle.com/drgilermo/nba-players-stats>`_ \
        NBA Players stats since 1950 dataset.

    ..note:: See \
    `Glossary <https://www.basketball-reference.com/about/glossary.html>`_ \
    for basketball statistic terms definitions.

    :Attributes:

    - **player**: *DataFrame* player dataset
    - **player_types**: *dict* data types for player dataset
    - **stats**: *DataFrame* season statistics dataset
    - **stats_types**: *dict* data types for season statistics dataset
    """
    def __init__(self):
        self.fame = None
        self.fame_types = {

        }
        self.players = None
        self.players_types = {
            'idx': np.int,
            'player': str,                                  # Player
            'height': np.float,                             # height
            'weight': np.float,                             # weight
            'collage': 'category',                          # collage
            'born': str,                                    # born
            'birth_city': str,                              # birth_city
            'birth_state': 'category',                      # birth_state
        }
        self.stats = None
        self.stats_types = {
            'idx': np.int,
            'season': str,                                  # Year
            'player': str,                                  # Player
            'position': 'category',                         # Pos
            'age': np.float,                                # Age
            'team': 'category',                             # Tm
            'games': np.int,                                # G
            'games_started': np.float,                      # GS
            'minutes_played': np.float,                     # MP
            'efficiency_rating': np.float,                  # PER
            'true_shooting_pct': np.float,                  # TS%
            '3_point_average_attempts': np.float,           # 3PAr
            'ftr': np.float,                                # FTr
            'offensive_rebound_pct': np.float,              # ORB%
            'defensive_rebound_pct': np.float,              # DRB%
            'total_rebound_pct': np.float,                  # TRB%
            'assist_pct': np.float,                         # AST%
            'steal_pct': np.float,                          # STL%
            'block_pct': np.float,                          # BLK%
            'turnovers_pct': np.float,                      # TOV%
            'usage_pct': np.float,                          # USG%
            'blank_1': str,                                 # Blank Line 1
            'offensive_win_shares': np.float,               # OWS
            'defensive_win_shares': np.float,               # DWS
            'win_shares': np.float,                         # WS
            'win_shares_48': np.float,                      # WS/48
            'blank_2': str,                                 # Blanks Line 2
            'offensive_box_plus_minus': np.float,           # OBPM
            'defensive_box_plus_minus': np.float,           # DBPM
            'box_plus_minus': np.float,                     # BPM
            'value_over_replacement_player': np.float,      # VORP
            'field_goals': np.int,                          # FG
            'field_goal_attempts': np.int,                  # FGA
            'field_goal_pct': np.float,                     # FG%
            '3_pointers': np.float,                         # 3P
            '3_painters_attempts': np.float,                # 3PA
            '3_pointers_pct': np.float,                     # 3P%
            '2_pointers': np.int,                           # 2P
            '2_pointers_attempts': np.int,                  # 2PA
            '2_pointers_pct': np.float,                     # 2P%
            'effective_field_goal_pct': np.float,           # eFG%
            'free_throws': np.int,                          # FT
            'free_throw_attempts': np.int,                  # FTA
            'free_throw_pct': np.float,                     # FT%
            'offensive_rebounds': np.float,                 # ORB
            'defensive_rebounds': np.float,                 # DRB
            'total_rebounds': np.float,                     # TRB
            'assists': np.float,                            # AST
            'steals': np.float,                             # STL
            'blocks': np.float,                             # BLK
            'turnovers': np.float,                          # TOV
            'fouls': np.int,                                # PF
            'points': np.int,                               # PTS
        }

        self.load_data()

    def __repr__(self):
        return 'Statistics()'

    def load_data(self):
        """
        Load the player and stats datasets.
        """
        def blank_filter(text):
            """
            Remove blank line entries.

            :param str text: original text
            :return: text with blank entries removed
            :rtype: str
            """
            return re.sub(r'^\d+,+$', '', text, flags=re.MULTILINE)

        def year_parser(year):
            """
            Format year string to be ISO 8601 string.
            :param year: year
            :return: year in ISO 8601 format
            :rtype: str
            """
            return [f'{x}-01-01' for x in year]

        self.players = (pd.read_csv(players_file,
                                    date_parser=year_parser,
                                    dtype=self.players_types,
                                    header=None,
                                    index_col=[5],
                                    names=self.players_types.keys(),
                                    parse_dates=[5],
                                    skiprows=1,
                                    )
                        .drop('idx', axis=1))
        logging.debug('Players Dataset Loaded')

        with open(season_file, 'r') as f:
            season_text = f.read()
            filtered_text = blank_filter(season_text)
            logging.info('Season Stats Dataset cleaned')

        self.stats = (pd.read_csv(io.StringIO(filtered_text),
                                  date_parser=year_parser,
                                  dtype=self.stats_types,
                                  header=None,
                                  index_col=[1],
                                  names=self.stats_types.keys(),
                                  parse_dates=[1],
                                  skiprows=1,
                                  )
                      .drop(['blank_1', 'blank_2', 'idx'], axis=1))
        logging.debug('Season Stats Dataset Loaded')


