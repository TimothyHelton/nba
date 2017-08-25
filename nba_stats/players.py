#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Players Module

.. moduleauthor:: Timothy Helton <timothy.j.helton@gmail.com>
"""
from collections import namedtuple
import io
import logging
import os.path as osp
import re
import urllib

from bokeh import io as bkio
from bokeh import models as bkm
from bs4 import BeautifulSoup
import geocoder
import matplotlib.pyplot as plt
import matplotlib.colors as mplcol
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import sklearn.decomposition as skdecomp
import sklearn.discriminant_analysis as skdisc
import sklearn.linear_model as sklinmod
import sklearn.metrics as skmetric
import sklearn.naive_bayes as sknb
import sklearn.preprocessing as skpre

from nba_stats.utils import save_fig, size
try:
    from nba_stats import keys
except ModuleNotFoundError:
    print('A Google API Key is required to generate the geographic images.')
    print('Upon instancing the Statistics class please assign your key to the '
          'google_key attribute.')


log_format = ('%(asctime)s  %(levelname)8s  -> %(name)s <- '
              '(line: %(lineno)d) %(message)s\n')
date_format = '%m/%d/%Y %I:%M:%S'
logging.basicConfig(format=log_format, datefmt=date_format,
                    level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

current_dir = osp.dirname(osp.realpath(__file__))
data_dir = osp.realpath(osp.join(current_dir, '..', 'data'))
players_file = osp.join(data_dir, 'Players.csv')
season_file = osp.join(data_dir, 'Seasons_Stats.csv')


def confusion_plot(matrix, save=False):
    """
    Create confusion matrix table.

    :param DataFrame matrix: confusion matrix to be plotted
    :param bool save: if True the figure will be saved
    """
    plt.figure('Confusion Matrix', figsize=(3, 3),
               facecolor='white', edgecolor='black')
    rows, cols = (1, 1)
    ax0 = plt.subplot2grid((rows, cols), (0, 0))

    labels = ['False', 'True']
    matrix.index = labels
    matrix.columns = labels
    matrix_max = matrix.values.max()

    cmap = mplcol.LinearSegmentedColormap.from_list(
        'white_blue', ['white'] * 2 + ['C0'])

    sns.heatmap(matrix, alpha=0.5, annot=True,
                annot_kws={'size': 14},
                cmap=cmap, cbar=False, fmt='.0f', linecolor='lightgray',
                linewidths=1, vmin=0, vmax=matrix_max, ax=ax0)

    ax0.xaxis.tick_top()
    ax0.set_xticklabels(ax0.xaxis.get_majorticklabels(),
                        fontsize=size['label'])
    ax0.set_yticklabels(ax0.yaxis.get_majorticklabels(),
                        fontsize=size['label'], rotation=0)

    super_title = plt.suptitle('Confusion Matrix',
                               fontsize=size['super_title'], x=0.38, y=1.12)

    save_fig('confusion_matrix', save, super_title)


Classify = namedtuple(
    'Classify', ['classify_report', 'confusion', 'model',
                 'score_test', 'score_train']
)

Subset = namedtuple(
    'Subset', ['data', 'feature_names', 'players',
               'x_test', 'x_train', 'y_test', 'y_train']
)

KPCA = namedtuple(
    'KPCA', ['feature_names', 'fit', 'model', 'n_components',
             'players', 'subset',
             'x_test', 'x_train', 'y_test', 'y_train']
)

PCA = namedtuple(
    'PCA', ['cut_off', 'feature_names', 'fit', 'model', 'n_components',
            'players', 'subset', 'var_pct', 'var_pct_cum', 'variance',
            'x_test', 'x_train', 'y_test', 'y_train']
)


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

    - **classify**: *namedtuple* classification model
        - fields:
            - classify_report
            - confusion
            - model
            - score_test
            - score_train
    - **evaluate**: *dict* model test and train scores
    - **fame**: *Series* players in the Hall of Fame
    - **features**: *DataFrame* model features
    - **feature_counts**: *list* quantities of samples with complete data for \
        a given subset of features
    - **feature_subsets**: *dict* subsets of the data created which have \
        all features defined in a namedtulple
        - fields:
            - data
            - feature_names
            - x_test
            - x_train
            - y_test
            - y_train
    - **google_key**: *str* Google API Key
    - **hof_birth_locations**: *DataFrame* male Hall of Fame birth locations, \
        latitude and longitude
    - **kernel_pca**: *dict* kernel principal component analysis
        - keys: *int* number of features
        - values: *tuple* (train_score, test_score)
    - **optimal_model**: *DataFrame* optimal model and feature quantity based \
        on test score
    - **pca**: *dict* principal component analysis
        - keys: *int* number of features
        - values: *tuple* (features, fit, transform, n_components, var_pct, \
            var_pct_cum, variance, cut_off, subset)
    - **players**: *DataFrame* player dataset
    - **players_fame**: *DataFrame* player dataset filtered to only include \
        Hall of Fame members
    - **players_types**: *dict* data types for player dataset
    - **seed**: *int* random seed value for samples
    - **stats**: *DataFrame* season statistics dataset
    - **stats_fame**: *DataFrame* stats dataset filtered to only include \
        Hall of Fame members
    - **stats_types**: *dict* data types for season statistics dataset
    """
    def __init__(self):
        self.classify = {}
        self.evaluate = None
        self.fame_types = {
            'name': str,
            'category': 'category'
        }
        try:
            self.fame = pd.read_csv('https://timothyhelton.github.io/'
                                    'assets/data/NBA_Hall_of_Fame.csv',
                                    dtype=self.fame_types,
                                    header=None,
                                    index_col=0,
                                    names=self.fame_types.keys(),
                                    skiprows=1,
                                    )
            logger.info('NBA Hall of Fame Players from '
                        'https://timothyhelton.github.io')
        except urllib.error.HTTPError:
            self.scrape_hall_of_fame()
        # Dan Issel's name is misspelled on the NBA Hall of Fame website
        self.fame.name = self.fame.name.str.replace('Dan Issell', 'Dan Issel')
        # Charles Cooper is listed as Chuck Cooper in the dataset
        self.fame.name = self.fame.name.str.replace('Charles Cooper',
                                                    'Chuck Cooper')
        # Richard Guerin's name is misspelled on the NBA Hall of Fame website
        self.fame.name = self.fame.name.str.replace('Richard Geurin',
                                                    'Richie Guerin')

        self.features = None
        self.feature_counts = None
        self.feature_subset = {}

        try:
            self.google_key = keys.GOOGLE_API_KEY
        except NameError:
            self.google_key = None

        self.hof_birth_locations = None
        self.kernel_pca = {}
        self.optimal_model = None
        self.pca = {}

        self.players = None
        self.players_types = {
            'idx': np.int,
            'player': str,                                  # Player
            'height': np.float,                             # height
            'weight': np.float,                             # weight
            'college': 'category',                          # college
            'born': str,                                    # born
            'birth_city': str,                              # birth_city
            'birth_state': 'category',                      # birth_state
        }
        self.players_fame = None

        self.seed = 0

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
        self.stats_fame = None

        self.training_size = 500
        self.x_test = None
        self.x_train = None
        self.y_test = None
        self.y_train = None

        self.load_data()
        self.get_feature_subsets()
        self.get_pca()

        try:
            self.hof_birth_locations = pd.read_csv(
                'https://timothyhelton.github.io/assets/data'
                '/hof_birth_locations.csv',
                index_col=0,
            )
            logger.info('Loaded Hall of Fame Birth Locations from '
                        'https://timothyhelton.github.io/assets/data/'
                        'hof_birth_locations.csv')
        except urllib.error.HTTPError:
            self.get_hof_birth_locations()

    def __repr__(self):
        return 'Statistics()'

    def classify_players(self, data, model='LR'):
        """
        Classify Data

        :param namedtuple data: object
        :param str model: model designator (see table below for \
            implemented types)

        +------------------+-------------------------------+
        | Model Designator | Scikit-Learn Model Type       |
        +==================+===============================+
        | LDA              | LinearDiscriminantAnalysis    |
        +------------------+-------------------------------+
        | LR               | LogisticRegression            |
        +------------------+-------------------------------+
        | NB               | GaussianNB                    |
        +------------------+-------------------------------+
        | QDA              | QuadraticDiscriminantAnalysis |
        +------------------+-------------------------------+
        """
        models = {
            'LDA': skdisc.LinearDiscriminantAnalysis(),
            'LR': sklinmod.LogisticRegression(),
            'NB': sknb.GaussianNB(),
            'QDA': skdisc.QuadraticDiscriminantAnalysis(),
        }

        if model not in models.keys():
            logger.error(f'Requested model {model} has not been implemented.')

        classify = (models[model]
                    .fit(data.x_train, data.y_train))
        score_train = classify.score(data.x_train, data.y_train)
        score_test = classify.score(data.x_test, data.y_test)

        predict = classify.predict(data.x_test)
        confusion = pd.DataFrame((skmetric
                                  .confusion_matrix(data.y_test, predict)))
        classify_report = (skmetric
                           .classification_report(data.y_test, predict))
        features = len(data.feature_names) - 1
        self.classify[features] = Classify(
            classify_report, confusion, classify, score_test, score_train)

    def evaluate_classification(self):
        """
        Evaluate all available models on all subsets of data.
        """
        self.get_pca()

        models = {'LDA': [], 'LR': [], 'NB': [], 'QDA': []}
        for features in self.pca:
            for model in models.keys():
                self.classify_players(self.pca[features], model=model)
                score_test = self.classify[features].score_test
                score_train = self.classify[features].score_train
                models[model].extend([score_test, score_train])

        iterables = [self.pca.keys(), ['test', 'train']]
        column_idx = pd.MultiIndex.from_product(iterables,
                                                names=['features', 'score'])
        self.evaluate = pd.DataFrame(list(models.values()),
                                     index=models.keys(), columns=column_idx)

    def evaluate_models(self, evaluations=10):
        """
        Determine the model and features that yield highest test score.

        :param int evaluations: number of evaluation cycles to perform
        """
        self.seed = None
        optimal_features = []
        for n in range(int(evaluations)):
            self.get_feature_subsets()
            logger.info(f'Evaluation:\t{n + 1:3.0f} / {evaluations}')
            self.evaluate_classification()
            max_sample = (self.evaluate
                          .xs('test', level='score', axis=1)
                          .max(axis=1))
            model = max_sample.argmax()
            max_model = self.evaluate.loc[model][:, 'test']
            features = (max_model
                        .loc[max_model == max_sample.max()]
                        .argmax())
            optimal_features.append([model, features])
        self.optimal_model = pd.DataFrame(optimal_features,
                                          columns=['model', 'features'])

    def evaluate_all_players(self, feature_qty=47, model='LR'):
        """
        Evaluate data against the model created with given number of features.

        ..note:: The **feature_qty** must be a number from the attribute \
            **feature_counts**.

        :param int feature_qty: number of features to include in model
        :param str model: classification model
        """
        data = self.pca[feature_qty]
        x_test = (skpre.StandardScaler()
                  .fit_transform(data.subset.drop('response', axis=1)))
        y_test = data.subset.response

        all_players = PCA(data.cut_off, data.feature_names, data.fit,
                          data.model, data.n_components, data.players,
                          data.subset, data.var_pct, data.var_pct_cum,
                          data.variance, x_test, data.x_train, y_test,
                          data.y_train)
        self.classify_players(all_players, model='LR')

        print(f'\n\nAll Players Dataset: {feature_qty} Features')
        print(f'Mean Score: {self.classify[feature_qty].score_test:.3f}')
        print(self.classify[feature_qty].classify_report)
        print('Confusion Matrix')
        print(self.classify[feature_qty].confusion)

    def evaluation_plot(self, save=False):
        """
        Evaluation heat map of classification models.

        :param bool save: if True the figure will be saved
        """
        self.evaluate_classification()

        plt.figure('Evaluation Heatmap', figsize=(3, 10),
                   facecolor='white', edgecolor='black')
        rows, cols = (1, 1)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))

        data = self.evaluate.xs('test', level='score', axis=1)
        cut = 0.8
        color_mask = (data[(data > -cut) & (data < cut)]
                      .fillna(0)
                      .astype(bool))
        data = data.mask(color_mask)

        cmap = mplcol.LinearSegmentedColormap.from_list(
            'white_blue', ['white'] * 8 + ['C0'] * 2)

        sns.heatmap(data.T, annot=True, cmap=cmap,
                    cbar_kws={'orientation': 'vertical'}, fmt='.2f',
                    linecolor='lightgray', linewidths=0.1, vmin=0, vmax=1,
                    ax=ax0)

        cbar = ax0.collections[0].colorbar
        cbar.set_ticks([0, 0.5, 1])
        cbar.ax.tick_params(labelsize=size['label'])
        cbar.outline.set_linewidth(1)
        cbar.outline.set_edgecolor('lightgray')

        ax0.set_title('Test Validation Score (0.8 Threshold)',
                      fontsize=size['label'])
        ax0.set_ylabel('Number of Features', fontsize=size['label'])
        model_names = ['Linear Discriminant Analysis',
                       'Logistic Regression',
                       'Naive Bayes',
                       'Quadratic Discriminant Analysis'
                       ]
        ax0.set_xticklabels(model_names,
                            fontsize=size['label'], rotation=90)
        ax0.set_yticklabels(ax0.yaxis.get_majorticklabels(),
                            fontsize=size['label'], rotation=0)

        super_title = plt.suptitle('Classification Model Comparision',
                                   fontsize=size['super_title'],
                                   x=0.5, y=0.95)

        save_fig('model_evaluation', save, super_title)

    def get_feature_subsets(self):
        """
        Split data on number of complete season statistics samples.
        """
        record_counts = self.features.count().sort_values().unique()

        for count in record_counts:
            data = (self.features
                    .loc[:, self.features.count() >= count]
                    .dropna())
            players = (self.stats
                       .loc[:, self.stats.count() >= count]
                       .dropna()
                       .player)
            feature_names = data.columns
            self.get_test_train(data)
            self.feature_subset[len(feature_names)] = Subset(
                data, feature_names, players,
                self.x_test, self.x_train, self.y_test, self.y_train)

        self.feature_counts = self.feature_subset.keys()

    def get_hof_birth_locations(self):
        """
        Get male Hall of Fame birth location latitude and longitude values.
        """
        locations_qty = (self.players_fame
                         .groupby('birth_state')
                         .player
                         .count()
                         .sort_values(ascending=False))
        locations_qty = locations_qty.iloc[locations_qty.nonzero()]

        locations = (self.players_fame
                     .birth_state
                     .unique()
                     .dropna())

        logger.info('Acquiring Hall of Fame birth geo coordinates')
        coordinates = []
        for place in locations:
            g = geocoder.google(place)
            lat, long = g.latlng
            coordinates.append([place, lat, long])

        self.hof_birth_locations = (pd.DataFrame(coordinates,
                                                 columns=['locations',
                                                          'latitude',
                                                          'longitude'])
                                    .set_index('locations'))
        self.hof_birth_locations['qty'] = locations_qty
        self.hof_birth_locations = (self.hof_birth_locations
                                    .reset_index()
                                    .sort_values(by=['qty', 'locations'],
                                                 ascending=[False, True])
                                    .set_index('locations'))
        logger.debug('Hall of Fame birth locations loaded')

    def get_kernel_pca(self, kernel='linear', n_components=10):
        """
        Perform Principal Component Analysis (PCA) using the kernel method.

        ..warning:: This is a computationally expensive method.

        :param str kernel: type of kernel to employee
        :param int n_components: number of principal components
        """
        for n, count in enumerate(self.feature_subset):
            subset = self.feature_subset[count]
            kpca = skdecomp.KernelPCA(kernel=kernel, n_components=n_components,
                                      n_jobs=-1)
            logger.debug(f'Calculating KPCA Subset: '
                         f'{n + 1} of {len(self.feature_subset)}')
            fit = kpca.fit(subset.x_test, subset.y_test)
            x_train_kpca = kpca.fit_transform(subset.x_train)
            x_test_kpca = kpca.transform(subset.x_test)

            self.kernel_pca[count] = KPCA(
                subset.feature_names, fit, kpca, n_components, subset.players,
                subset.data, x_test_kpca, x_train_kpca, subset.y_test,
                subset.y_train)

        logger.debug('Kernel PCA Complete')

    def get_pca(self):
        """
        Perform Principal Component Analysis (PCA).
        """
        for count in self.feature_counts:
            subset = self.feature_subset[count]
            pca = skdecomp.PCA()
            fit = pca.fit(subset.x_train, subset.y_train)
            x_train_pca = pca.fit_transform(subset.x_train)
            x_test_pca = pca.transform(subset.x_test)
            n_components = pca.n_components_

            var_pct = fit.explained_variance_ratio_
            var_pct_cum = var_pct.cumsum()
            variance = pd.DataFrame(np.c_[var_pct, var_pct_cum],
                                    columns=['var_pct', 'var_pct_cum'])
            ddf_var_pct = variance.var_pct.diff().diff()
            cut_off = ddf_var_pct[ddf_var_pct < 0].index.tolist()[0]

            self.pca[n_components] = PCA(
                cut_off, subset.feature_names, fit, pca,
                subset.players, n_components, subset.data, var_pct,
                var_pct_cum, variance, x_test_pca, x_train_pca, subset.y_test,
                subset.y_train)

        logger.debug('PCA Complete')

    def get_test_train(self, data, test_split=0.2):
        """
        Get balanced test and training datasets by bootstrapping.

        :param DataFrame data: data to be partitioned
        :param float test_split: test set percentage
        """
        fame = data.query('response == 1')
        regular = data.query('response == 0')

        fame_test = fame.sample(frac=test_split, random_state=self.seed,
                                replace=False)
        fame_train = fame_test[~fame_test.duplicated(keep=False)]

        regular_test = regular.sample(n=fame_test.shape[0],
                                      random_state=self.seed, replace=False)
        regular_train = regular_test[~regular_test.duplicated(keep=False)]

        fame_boot = fame_train.sample(n=self.training_size,
                                      random_state=self.seed, replace=True)
        regular_boot = regular_train.sample(n=self.training_size,
                                            random_state=self.seed,
                                            replace=True)

        bootstrap = (pd.concat([fame_boot, regular_boot])
                     .sample(frac=1, random_state=self.seed)
                     .reset_index(drop=True))

        test = (pd.concat([fame_test, regular_test])
                .sample(frac=1, random_state=self.seed)
                .reset_index(drop=True))

        self.x_test = (skpre.StandardScaler()
                       .fit_transform(test.drop('response', axis=1)))
        self.x_train = (skpre.StandardScaler()
                        .fit_transform(bootstrap.drop('response', axis=1)))
        self.y_test = test.response
        self.y_train = bootstrap.response

    def hof_birth_loc_plot(self, save=False):
        """
        Horizontal Bar chart of birth locations for male Hall of Fame players.

        :param bool save: if True the figure will be saved
        """
        plt.figure('Hall of Fame Birth Locations', figsize=(14, 10),
                   facecolor='white', edgecolor=None)
        rows, cols = (1, 1)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))

        locations = (self.hof_birth_locations
                     .qty
                     .sort_values(ascending=True))
        (locations
         .plot(kind='barh', alpha=0.5, color=['gray'],
               edgecolor='black', legend=None, width=0.7, ax=ax0))

        medium = (locations[(locations > 4) & (locations < 10)]
                  .index
                  .get_values())
        for patch in medium:
            position = locations.index.get_loc(patch)
            ax0.patches[position].set_facecolor('C0')
            ax0.patches[position].set_alpha(0.5)

        high = (locations[locations >= 10]
                .index
                .get_values())
        for patch in high:
            position = locations.index.get_loc(patch)
            ax0.patches[position].set_facecolor('C0')
            ax0.patches[position].set_alpha(0.7)

        for patch in ax0.patches:
            width = patch.get_width()
            ax0.text(x=width - 0.1,
                     y=patch.get_y() + 0.1,
                     s=f'{width:.0f}',
                     ha='right')

        ax0.set_title('Birth Locations', fontsize=size['title'])
        ax0.set_ylabel('')

        ax0.set_xticklabels('')
        ax0.xaxis.set_ticks_position('none')
        ax0.yaxis.set_ticks_position('none')
        ax0.set_yticklabels(ax0.yaxis.get_majorticklabels(),
                            fontsize=size['legend'])

        for side in ('top', 'right', 'bottom', 'left'):
            ax0.spines[side].set_visible(False)

        super_title = plt.suptitle('Hall of Fame Players',
                                   fontsize=size['super_title'],
                                   x=0.05, y=0.95)

        save_fig('hof_birth_locations', save, super_title)
        logger.debug('Create Hall of Fame Birth Locations Plot')

    def hof_birth_map_plot(self):
        """
        Plot male player Hall of Fame birth locations.

        .. warning:: This method requires a Google API Key
        """
        map_options = {
            'lat': 39.50,
            'lng': -98.35,
            'map_type': 'roadmap',
            'zoom': 4,
        }
        plot = bkm.GMapPlot(
            api_key=keys.GOOGLE_API_KEY,
            x_range=bkm.Range1d(),
            y_range=bkm.Range1d(),
            map_options=bkm.GMapOptions(**map_options),
            plot_width=800,
            plot_height=600,
        )
        plot.title.text = 'Hall of Fame Birth Locations'

        location = bkm.Circle(
            x='longitude',
            y='latitude',
            fill_alpha=0.8,
            fill_color='#cc0000',
            line_color=None,
            size='m_size',
        )

        self.hof_birth_locations['m_size'] = self.hof_birth_locations.qty + 5
        locations = bkm.sources.ColumnDataSource(self.hof_birth_locations)
        plot.add_glyph(locations, location)

        hover = bkm.HoverTool()
        hover.tooltips = [
            ('Hall of Fame Players', '@qty'),
        ]
        plot.add_tools(
            hover,
            bkm.PanTool(),
            bkm.WheelZoomTool(),
        )

        bkio.output_file('hof_birth_map.html')
        bkio.show(plot)

    def hof_category_plot(self, save=False):
        """
        Horizontal Bar chart of Hall of Fame categories.

        :param bool save: if True the figure will be saved
        """
        plt.figure('Hall of Fame Categories', figsize=(12, 3),
                   facecolor='white', edgecolor=None)
        rows, cols = (1, 1)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))

        categories = (self.fame
                      .groupby('category')
                      .count()
                      .sort_values(by='name'))

        categories.plot(kind='barh', alpha=0.5, color=['gray'],
                        edgecolor='black', legend=None, width=0.7, ax=ax0)

        emphasis = categories.index.get_loc('Player')
        ax0.patches[emphasis].set_facecolor('C0')
        ax0.patches[emphasis].set_alpha(0.7)

        for patch in ax0.patches:
            width = patch.get_width()
            ax0.text(x=width - 1,
                     y=patch.get_y() + 0.17,
                     s=f'{width:.0f}',
                     fontsize=size['label'],
                     ha='right')

        ax0.set_title('Categories', fontsize=size['title'])
        ax0.set_ylabel('')

        ax0.set_xticklabels('')
        ax0.xaxis.set_ticks_position('none')
        ax0.yaxis.set_ticks_position('none')
        ax0.set_yticklabels(ax0.yaxis.get_majorticklabels(),
                            fontsize=size['legend'])

        for side in ('top', 'right', 'bottom', 'left'):
            ax0.spines[side].set_visible(False)

        super_title = plt.suptitle('Hall of Fame Players',
                                   fontsize=size['super_title'],
                                   x=0.05, y=1.09)

        save_fig('hof_category', save, super_title)
        logger.debug('Create Hall of Fame Category Plot')

    def hof_college_plot(self, save=False):
        """
        Horizontal Bar chart of Hall of Fame College Attendance.

        :param bool save: if True the figure will be saved
        """
        plt.figure('Hall of Fame Attended College', figsize=(8, 1),
                   facecolor='white', edgecolor=None)
        rows, cols = (1, 1)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))

        yes = self.players_fame.college.count()
        no = self.players_fame.college.size - yes

        attend = pd.Series([yes, no], index=['Yes', 'No']).sort_index()

        attend.plot(kind='barh', alpha=0.5, color=['gray'],
                    edgecolor='black', legend=None, width=0.7, ax=ax0)

        emphasis = attend.index.get_loc('Yes')
        ax0.patches[emphasis].set_facecolor('C0')
        ax0.patches[emphasis].set_alpha(0.7)

        for patch in ax0.patches:
            width = patch.get_width()
            ax0.text(x=width - 1,
                     y=patch.get_y() + 0.16,
                     s=f'{width:.0f}',
                     fontsize=size['label'],
                     ha='right')

        ax0.set_title('Attended College',
                      fontsize=size['title'])
        ax0.set_ylabel('')

        ax0.set_xticklabels('')
        ax0.xaxis.set_ticks_position('none')
        ax0.yaxis.set_ticks_position('none')
        ax0.set_yticklabels(ax0.yaxis.get_majorticklabels(),
                            fontsize=size['legend'])

        for side in ('top', 'right', 'bottom', 'left'):
            ax0.spines[side].set_visible(False)

        super_title = plt.suptitle('Hall of Fame Players',
                                   fontsize=size['super_title'],
                                   x=0.15, y=1.6)

        save_fig('hof_college', save, super_title)
        logger.debug('Create Hall of Fame College Attendance Plot')

    def hof_correlation_plot(self, save=False):
        """
        Correlation heat map of Hall of Fame statistic features.

        :param bool save: if True the figure will be saved
        """
        plt.figure('Correlation Heatmap', figsize=(16, 14),
                   facecolor='white', edgecolor='black')
        rows, cols = (1, 1)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))

        correlation = self.stats_fame.corr()
        cut = 0.5
        color_mask = (correlation[(correlation > -cut) & (correlation < cut)]
                      .fillna(0)
                      .astype(bool))
        correlation[color_mask] = 0

        cmap = mplcol.LinearSegmentedColormap.from_list(
            'blue_white_blue', ['indianred'] + ['white'] * 3 + ['C0'])

        sns.heatmap(correlation, center=0,
                    cmap=cmap,
                    cbar_kws={'orientation': 'vertical'},
                    linecolor='lightgray', linewidths=0.1, vmin=-1, vmax=1,
                    ax=ax0)

        cbar = ax0.collections[0].colorbar
        cbar.set_ticks(np.arange(-1, 1.5, 0.5).tolist())
        cbar.ax.tick_params(labelsize=size['label'])
        cbar.outline.set_linewidth(1)
        cbar.outline.set_edgecolor('lightgray')

        ax0.set_title('Statistics Correlation (0.5 Threshold)',
                      fontsize=size['title'])
        ax0.set_xticklabels(ax0.xaxis.get_majorticklabels(),
                            fontsize=size['label'], rotation=90)
        ax0.set_yticklabels(ax0.yaxis.get_majorticklabels(),
                            fontsize=size['label'], rotation=0)

        super_title = plt.suptitle('Hall of Fame Players',
                                   fontsize=size['super_title'],
                                   x=0.03, y=0.93)

        save_fig('hof_correlation', save, super_title)

    def hof_percent_plot(self, save=False):
        """
        Large percent image of players to make it to the Hall of Fame.

        :param bool save: if True the figure will be saved
        """
        plt.figure('Hall of Fame Percentage', figsize=(6, 2.1),
                   facecolor='white', edgecolor=None)
        rows, cols = (1, 1)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))

        hof_qty = self.players_fame.shape[0]
        all_qty = self.players.shape[0]
        hof_pct = hof_qty / all_qty * 100
        ax0.text(x=0, y=0, s=f'{hof_pct:.1f}%', color='C0', fontsize=125,
                 ha='left', va='bottom')
        ax0.text(x=0, y=0, s='NBA Players in the Hall of Fame', color='C0',
                 fontsize=20, ha='left')

        ax0.axis('off')

        save_fig('hof_percent', save)
        logger.debug('Create Hall of Fame Percent Plot')

    def hof_player_breakdown_plot(self, save=False):
        """
        Horizontal bar plot of Hall of Fame player category breakdown.

        :param bool save: if True the figure will be saved
        """
        plt.figure('Hall of Fame Player Subcategories', figsize=(12, 3),
                   facecolor='white', edgecolor=None)
        rows, cols = (1, 1)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))

        total = self.fame.query('category == "Player"').shape[0]
        nba = self.players_fame.shape[0]
        categories = (pd.DataFrame([total, nba, 42, 16],
                                   index=pd.Index(data=['Total', 'NBA',
                                                        'Non-NBA Men',
                                                        'Women'],
                                                  name='Hall of Fame'),
                                   columns=['inductees'])
                      .sort_values(by='inductees', ascending=True))

        categories.plot(kind='barh', alpha=0.5, color=['gray'],
                        edgecolor='black', legend=None, width=0.7, ax=ax0)

        emphasis = categories.index.get_loc('NBA')
        ax0.patches[emphasis].set_facecolor('C0')
        ax0.patches[emphasis].set_alpha(0.7)

        for patch in ax0.patches:
            width = patch.get_width()
            ax0.text(x=width - 1,
                     y=patch.get_y() + 0.2,
                     s=f'{width:.0f}',
                     fontsize=size['label'],
                     ha='right')

        ax0.set_title('Player Subcategories', fontsize=size['title'])
        ax0.set_ylabel('')

        ax0.set_xticklabels('')
        ax0.xaxis.set_ticks_position('none')
        ax0.yaxis.set_ticks_position('none')
        ax0.set_yticklabels(ax0.yaxis.get_majorticklabels(),
                            fontsize=size['legend'])

        for side in ('top', 'right', 'bottom', 'left'):
            ax0.spines[side].set_visible(False)

        super_title = plt.suptitle('Hall of Fame Players',
                                   fontsize=size['super_title'],
                                   x=0.05, y=1.09)

        save_fig('hof_player_subcategory', save, super_title)
        logger.debug('Create Hall of Fame Player Subcategory Plot')

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
                                    index_col=5,
                                    names=self.players_types.keys(),
                                    parse_dates=[5],
                                    skiprows=1,
                                    )
                        .drop('idx', axis=1)
                        .dropna(how='all'))
        self.players.player = self.players.player.str.replace('*', '')

        with open(season_file, 'r') as f:
            season_text = f.read()
            filtered_text = blank_filter(season_text)
            logger.info('Season Stats Dataset cleaned')

        self.stats = (pd.read_csv(io.StringIO(filtered_text),
                                  date_parser=year_parser,
                                  dtype=self.stats_types,
                                  header=None,
                                  index_col=1,
                                  names=self.stats_types.keys(),
                                  parse_dates=[1],
                                  skiprows=1,
                                  )
                      .drop(['blank_1', 'blank_2', 'idx'], axis=1))
        self.stats.player = self.stats.player.str.replace('*', '')

        # Hall of Fame
        filter_players = self.fame.query('category == "Player"').name

        self.players_fame = self.players[(self.players.player
                                          .isin(filter_players))]

        stats_mask = self.stats.player.isin(filter_players)
        self.stats_fame = self.stats[stats_mask]

        # Features and Response
        self.features = self.stats.drop('player', axis=1)
        for col in self.features.select_dtypes(include=['category']).columns:
            self.features[col] = (self.features[col]
                                  .cat
                                  .codes)
        self.features['response'] = 0
        self.features.loc[stats_mask, 'response'] = 1
        logger.info('Datasets Loaded')

    def missing_hall_of_fame(self):
        """
        Players in the Hall of Fame without entries in the datasets.

        :return: names of players not found in the Players and Season Stats \
            datasets
        :rtype: DataFrame
        """
        no_players = (self.fame[~self.fame.isin((self.players_fame
                                                 .player
                                                 .tolist()))]
                      .dropna()
                      .query('category == "Player"')
                      .name)
        logger.debug('DIFF players Hall of Fame to Players Dataset complete')
        no_stats = (self.fame[~self.fame.isin(self.stats.player.unique())]
                    .dropna()
                    .query('category == "Player"')
                    .name)
        logger.debug('DIFF players Hall of Fame to Stats Dataset complete')
        return (pd.concat([no_players, no_stats], axis=1, join='outer',
                          ignore_index=True)
                .rename(columns={n: name for n, name
                                 in enumerate(('player_dataset',
                                               'stats_dataset'))})
                .reset_index(drop=True))

    def optimal_features_plot(self, evaluations=100, save=False):
        """
        Histogram of optimal feature distribution.

        :param int evaluations: number of random evaluations
        :param bool save: if True the figure will be saved
        """
        self.evaluate_models(evaluations=evaluations)

        plt.figure('Optimal Features Distribution Plot', figsize=(6, 8),
                   facecolor='white', edgecolor='black')
        rows, cols = (1, 1)
        ax0 = plt.subplot2grid((rows, cols), (0, 0))

        data = (self.optimal_model
                .set_index('model')
                .features
                .value_counts()
                .sort_index())
        (data
         .plot(kind='bar', alpha=0.5, color='gray',
               edgecolor='black', legend=False, ax=ax0))

        max_height = 0
        tallest_patch = None
        for patch in ax0.patches:
            height = patch.get_height()
            if height > max_height:
                max_height = height
                tallest_patch = patch

        tallest_patch.set_facecolor('C0')
        tallest_patch.set_alpha(0.7)

        ax0.set_title(f'Generated Cycles: {evaluations}',
                      fontsize=size['title'])
        ax0.set_xlabel('Number of Model Features', fontsize=size['label'])
        ax0.set_xticklabels(ax0.xaxis.get_majorticklabels(), rotation=0)
        ax0.set_ylabel('Count', fontsize=size['label'])

        for side in ('top', 'right'):
            ax0.spines[side].set_visible(False)

        super_title = plt.suptitle('Optimal Model Features',
                                   fontsize=size['super_title'], x=0.35)

        save_fig('optimal_features', save, super_title)

    @staticmethod
    def pca_plot(pca, save=False):
        """
        Bar plot of Principle Components variance percentage.

        :param Series pca: principle component analysis class
        :param bool save: if True the figure will be saved
        """
        plt.figure('Hall of Fame PCA', figsize=(12, 8),
                   facecolor='white', edgecolor=None)
        rows, cols = (2, 2)
        ax0 = plt.subplot2grid((rows, cols), (0, 0), colspan=2)
        ax1 = plt.subplot2grid((rows, cols), (1, 0))
        ax2 = plt.subplot2grid((rows, cols), (1, 1))

        (pca.variance.iloc[:pca.cut_off]
         .rename(index={x: x + 1 for x in range(pca.var_pct.size)})
         .plot(kind='bar', alpha=0.5, color=['C0', 'gray'], edgecolor='black',
               width=0.7, ax=ax0))

        ax0.set_title('Percent of Variance Explained', fontsize=size['title'])
        ax0.legend(['Individual Variance', 'Cumulative Variance'],
                   frameon=False, fontsize=size['legend'])
        legend_handles = ax0.get_legend().legendHandles
        legend_handles[0].set_alpha(0.7)
        legend_handles[0].set_edgecolor('black')
        ax0.set_xlabel('Principal Components', fontsize=size['label'])
        ax0.set_xticklabels(ax0.xaxis.get_majorticklabels(), rotation=0)

        for patch in range(pca.cut_off):
            emphasis = pca.variance.index.get_loc(patch)
            ax0.patches[emphasis].set_facecolor('C0')
            ax0.patches[emphasis].set_alpha(0.7)

        for patch in ax0.patches:
            height = patch.get_height()
            ax0.text(x=patch.get_x() + patch.get_width() / 2,
                     y=height - 0.035,
                     s=f'{height * 100:1.0f}%',
                     ha='center')

        fame = pca.y_train.values.astype('bool')
        reg = np.invert(fame)

        ax1.scatter(x=pca.x_train[reg][:, 0], y=pca.x_train[reg][:, 1],
                    alpha=0.5, color='gray', label='Regular Players',
                    marker='o')
        ax1.scatter(x=pca.x_train[fame][:, 0],
                    y=pca.x_train[fame][:, 1],
                    alpha=0.7, color='C0', label='Hall of Fame', marker='d')

        ax1.set_title('$2^{nd}$ vs $1^{st}$ Principal Component',
                      fontsize=size['title'])
        ax1.set_xlabel('$1^{st}$ Principal Component', fontsize=size['label'])
        ax1.set_ylabel('$2^{nd}$ Principal Component', fontsize=size['label'])

        ax2.scatter(x=pca.x_train[reg][:, 1], y=pca.x_train[reg][:, 2],
                    alpha=0.5, color='gray', label='Regular Players',
                    marker='o')
        ax2.scatter(x=pca.x_train[fame][:, 1],
                    y=pca.x_train[fame][:, 2],
                    alpha=0.7, color='C0', label='Hall of Fame', marker='d')

        ax2.set_title('$3^{rd}$ vs $2^{nd}$ Principal Component',
                      fontsize=size['title'])
        ax2.legend(frameon=False)
        ax2.set_xlabel('$2^{nd}$ Principal Component', fontsize=size['label'])
        ax2.set_ylabel('$3^{rd}$ Principal Component', fontsize=size['label'])

        for side in ('bottom', 'left'):
            ax0.spines[side].set_visible(False)

        for ax in (ax1, ax2):
            legend = ax.legend()
            legend.set_alpha(0.5)
            ax.get_xaxis().set_ticks([])

        for ax in (ax0, ax1, ax2):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_yaxis().set_ticks([])

        plt.tight_layout()
        super_title = plt.suptitle('Principal Components Analysis Training '
                                   f'Model: {pca.n_components:3.0f} Features',
                                   fontsize=size['super_title'],
                                   x=0.41, y=1.05)

        save_fig('pca', save, super_title)

    def scrape_hall_of_fame(self):
        """
        Scrape all the NBA Hall of Fame inductees.

        ..note:: NBA Hall of Fame Inductees scraped from www.nba.com website.
        """
        url = ('http://www.nba.com/history/naismith-memorial-basketball-hall'
               '-of-fame-inductees/')
        request = requests.get(url)
        soup = BeautifulSoup(request.text, 'lxml')
        section = soup.find('section', id='nbaArticleContent')
        tags = section.find_all('p')
        members = re.findall(r'<p>\s<b>(.+?)</b>(.+?)</p>', str(tags))
        remove_tags = [(x[0], re.sub(r'</?\w>', '', x[1]))
                       for x in members[1:]]
        remove_spaces = [(x[0], re.sub(r'\s', '', x[1])) for x in remove_tags]
        remove_commas = [(x[0], re.sub(r',', '', x[1])) for x in remove_spaces]
        self.fame = pd.DataFrame(remove_commas, columns=['name', 'category'])
        logger.info('NBA Hall of Fame Players Scraped from www.nba.com')
