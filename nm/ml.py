import tpot
import numbers
import logging
import warnings
import numpy as np
import pandas as pd
from copy import copy
from tqdm import tqdm
import pickle5 as pickle
from functools import partial
from operator import itemgetter
from collections import Iterable
from nm.util import make_bak_file
from deap.gp import PrimitiveTree
from deap import tools, base, creator
from tpot.builtins import StackingEstimator
from nm.util import is_serializable, trim_run
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from tpot.operator_utils import TPOTOperatorClassFactory, Operator, ARGType

TPOT_DEFAULT_CONFIG_DICT = 'TPOT light'

YIELD = 'others_dr'

SYMBOL = 'symbol'

OPEN_TIME = 'date'

DEFAULT_IMP_METHOD = 'coef'

MODELS_FILENAME = 'models.dat'

NON_SERIALIZABLE = (
'_pop_', '_pareto_front', '_toolbox', '_pset', '_log_file', 'operators_context', 'log_file_', 'operators', 'arguments',
'warm_start', 'generations', '_pop', '_pbar')

RANDOM_STATE = 41

if not hasattr(creator, 'FitnessMulti'):
    creator.create(
            name="FitnessMulti",
            base=base.Fitness,
            weights=(-1.0, 1.0))

if not hasattr(creator, 'Individual'):
    creator.create(
            name="Individual",
            base=PrimitiveTree,
            fitness=creator.FitnessMulti,
            statistics=dict,
            )


class Regressor(tpot.TPOTRegressor):
    non_serializable = set(NON_SERIALIZABLE)

    def __getstate__(self):
        if hasattr(self, '_optimized_pipeline') and not isinstance(self._optimized_pipeline, str):
            self._optimized_pipeline = PrimitiveTree(self._optimized_pipeline).__str__()
        attributes = self.__dict__.copy()
        # for attr in [a for a in attributes.keys() if not is_serializable(attributes.get(a))]:
        for attr in attributes.keys():
            if attr in self.non_serializable:
                attributes[attr] = []
        return attributes

    def __setstate__(self, state):
        self.__dict__ = state
        rebuild_me(self)
        self._update_non_serializable()

    def fit(self, features, target, sample_weight=None, groups=None):
        super().fit(features, target, sample_weight=None, groups=None)
        self._update_non_serializable()

    def _update_non_serializable(self):
        attributes = self.__dict__.copy()
        self.non_serializable.union([a for a in attributes.keys() if not is_serializable(attributes.get(a))])


class Classifier(tpot.TPOTClassifier):
    non_serializable = set(NON_SERIALIZABLE)

    def __getstate__(self):
        if hasattr(self, '_optimized_pipeline') and not isinstance(self._optimized_pipeline, str):
            self._optimized_pipeline = PrimitiveTree(self._optimized_pipeline).__str__()
        attributes = self.__dict__.copy()
        # for attr in [a for a in attributes.keys() if not is_serializable(attributes.get(a))]:
        for attr in attributes.keys():
            if attr in self.non_serializable:
                attributes[attr] = []
        return attributes

    def __setstate__(self, state):
        self.__dict__ = state
        rebuild_me(self)
        self._update_non_serializable()

    def fit(self, features, target, sample_weight=None, groups=None):
        super().fit(features, target, sample_weight=None, groups=None)
        self._update_non_serializable()

    def _update_non_serializable(self):
        attributes = self.__dict__.copy()
        self.non_serializable.union([a for a in attributes.keys() if not is_serializable(attributes.get(a))])


def add_attr(fitness, items):
    for ix, i in enumerate(items):
        i.__str__ = partial(PrimitiveTree.__str__, i)
        i.fitness = fitness[ix]
    return [items[i] for i, _ in sorted(enumerate(fitness), key=itemgetter(1), reverse=True)]


def serialize_me(obj):
    obj = copy(obj)
    if hasattr(obj, '_optimized_pipeline') and not isinstance(obj._optimized_pipeline, str):
        obj._optimized_pipeline = PrimitiveTree(obj._optimized_pipeline).__str__()
    for a in vars(obj).keys():
        if hasattr(obj, a) and not is_serializable(getattr(obj, a)):
            setattr(obj, a, [])
    return obj


def rebuild_me(self):
    for a in NON_SERIALIZABLE:
        if not hasattr(self, a):
            setattr(self, a, [])
    self._setup_config(self.config_dict)
    self._setup_template(self.template)
    self.verbosity = 0
    self.warm_start = True
    for key in sorted(self._config_dict.keys()):
        op_class, arg_types = TPOTOperatorClassFactory(
                key,
                self._config_dict[key],
                BaseClass=Operator,
                ArgBaseClass=ARGType,
                verbose=self.verbosity,
                )
        if op_class:
            self.operators.append(op_class)
            self.arguments += arg_types
    self.operators_context = {
        "make_pipeline"      : make_pipeline,
        "make_union"         : make_union,
        "StackingEstimator"  : StackingEstimator,
        "FunctionTransformer": FunctionTransformer,
        "copy"               : copy,
        }
    setattr(self, '_pareto_front', tools.ParetoFront(similar=lambda ind1, ind2: np.allclose(
            ind1.fitness.values, ind2.fitness.values)))
    self._pbar = None
    self._setup_pset()
    self._setup_toolbox()
    self._pop = self._toolbox.population(n=self.population_size)
    # if not hasattr(creator, 'FitnessMulti'):
    #     creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
    # if not hasattr(creator, 'Individual'):
    #     creator.create(
    #             "Individual",
    #             PrimitiveTree,
    #             fitness=creator.FitnessMulti,
    #             statistics=dict,
    #             )
    if hasattr(self, 'pareto_front_fitted_pipelines_') and isinstance(self.pareto_front_fitted_pipelines_, dict):
        items = [creator.Individual(PrimitiveTree([]).from_string(i, self._pset)) for i in
            self.pareto_front_fitted_pipelines_.keys()]
        keys = [(lambda d: creator.FitnessMulti((d.get('operator_count', 0), d.get('internal_cv_score', 0))))(
                self.evaluated_individuals_.get(k, {})) for k in self.pareto_front_fitted_pipelines_.keys()]
    elif hasattr(self, 'evaluated_individuals_') and isinstance(self.evaluated_individuals_, dict):
        items = [creator.Individual(PrimitiveTree([]).from_string(i, self._pset)) for i in
            self.evaluated_individuals_.keys()]
        keys = [creator.FitnessMulti((d.get('operator_count', 0), d.get('internal_cv_score', 0))) for d in
            self.evaluated_individuals_.values()]
    else:
        self.warm_start = False
        self.verbosity = 3
        return
    items = add_attr(keys, items)
    setattr(self._pareto_front, 'items', items)
    setattr(self._pareto_front, 'keys', sorted(keys))
    if hasattr(self, '_optimized_pipeline') and isinstance(self._optimized_pipeline, str):
        optimized_pipeline = creator.Individual(
                PrimitiveTree([]).from_string(self._optimized_pipeline, self._pset))
        optimized_pipeline.__str__ = partial(PrimitiveTree.__str__, optimized_pipeline)
        keys = [creator.FitnessMulti((d.get('operator_count', 0), d.get('internal_cv_score', 0))) for k, d in
            self.evaluated_individuals_.items() if k == optimized_pipeline.__str__()]
        if len(keys) > 0:
            optimized_pipeline.fitness = keys[0]
        else:
            optimized_pipeline.fitness.values = (5000.0, -float("inf"))
        self._optimized_pipeline = optimized_pipeline

    setattr(self, '_last_optimized_pareto_front', [v for i in self._pareto_front.keys for v in i.values[-1:]])

    if not hasattr(self, '_last_optimized_pareto_front_n_gens'):
        if hasattr(self, 'evaluated_individuals_'):
            last_gen = max([d.get('generation') for d in list(self.evaluated_individuals_.values())])
        else:
            last_gen = 0
        setattr(self, '_last_optimized_pareto_front_n_gens', last_gen)
    else:
        last_gen = self._last_optimized_pareto_front_n_gens

    if not hasattr(self, 'evaluated_individuals_'):
        setattr(self, 'evaluated_individuals_', {p.__str__(): (
            lambda v: {'generation': last_gen, 'mutation_count': 0, 'crossover_count': 0, 'predecessor': ('ROOT',),
                'operator_count'   : v[0], 'internal_cv_score': v[-1]})(self._pareto_front.keys[i].values) for i, p in
            enumerate(self._pareto_front.items)})

    self.verbosity = 3
    return self


class MarketModels:

    def __init__(self, filename=None, load=True):
        self._filename = filename
        self._last_model_name = None
        if load:
            self._models = self.load_models()
        else:
            self._models = {}
        if len(self._models) > 0:
            self.activate([*self._models.keys()][-1])

    def __getattr__(self, attr):
        if attr[0] != '_' and not hasattr(super(), attr) and hasattr(self.model, attr):
            setattr(self, attr, partial(self.apply_mode, method_name=attr))
            return getattr(self, attr)
        super().__getattribute__(attr)

    # noinspection PyShadowingBuiltins
    def __repr__(self):
        repr = 'BitScreener.MarketModels current models:'
        for model_name, model_dict in self._models.items():
            model = model_dict.get("model")
            if hasattr(model, '_estimator_type'):
                repr += f'\n\t{model_name}: [{model._estimator_type}]'
            else:
                repr += f'\n\t{model_name}: [{type(model)}]'
        return repr

    def __str__(self):
        to_print = ''
        for model_name, model_dict in self._models.items():
            model = model_dict.get("model")
            if hasattr(model, '_estimator_type'):
                to_print += f'{model_name}: [{model._estimator_type}], '
            else:
                to_print += f'{model_name}: [{type(model)}]'
        return to_print

    @property
    def filename(self):
        if self._filename is None:
            try:
                # noinspection PyPep8Naming,PyShadowingNames
                from config import ml_models as MODELS_FILENAME
            except (ImportError, ModuleNotFoundError):
                global MODELS_FILENAME
            self._filename = MODELS_FILENAME
        return self._filename

    @property
    def features(self):
        features_dict = {}
        for model_name, model_dict in self._models.items():
            features = model_dict.get('features', [])
            features_dict.update({model_name: features})
        return features_dict if self.last_model is None else features_dict.get(self._last_model_name)

    @property
    def last_model(self):
        return self.models.get(self._last_model_name, {}).get('model')

    @property
    def active(self):
        return self._last_model_name is not None and len(self._last_model_name) > 1

    @property
    def model(self):
        if self._last_model_name is not None and len(self._last_model_name) > 0:
            return {k: (lambda model: model.fitted_pipeline_ if isinstance(model,
                    (Regressor, Classifier)) and hasattr(model, 'fitted_pipeline_') else model
                    )(m.get('model')) for k, m in self._models.items()}.get(self._last_model_name)
        else:
            return {k: m.get('model') for k, m in self._models.items()}

    @property
    def models(self):
        return self._models

    @property
    def target(self):
        targets = {}
        for model_name, model_dict in self._models.items():
            target = model_dict.get('target', [])
            targets.update({model_name: target})
        return targets if self.last_model is None else targets.get(self._last_model_name)

    @staticmethod
    def trim_XY(x, y=None):
        max_value = x[x < np.inf].max()
        min_value = x[x > -np.inf].min()
        x[x == np.inf] = max_value
        x[x == -np.inf] = min_value
        x = np.nan_to_num(x)
        if y is not None:
            y = np.nan_to_num(y)
        return x if y is None else (x, y)

    @staticmethod
    def valid_as_features_columns(df, target=None, exclude=()):
        columns = [c for c in df.columns if df[c].dtype in (np.int, np.float, np.bool) and c not in [target, *exclude]]
        return columns

    @staticmethod
    def target_imputation(df, target):
        if target is None:
            target = YIELD
        if target == YIELD:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df2 = df.reset_index().sort_values([SYMBOL, OPEN_TIME])
                df2[target] = ((df2.Close - df2.Open) / df2.Open * 100).shift(-1)
                df2.loc[df2[df2[SYMBOL] != df2[SYMBOL].shift(-1)][YIELD].index, YIELD] = np.NaN
                df_index_name = df.index.name
                df = df.reset_index()
                df[target] = df2[target]
                df = df.set_index('index' if df_index_name is None else df_index_name)
        return df, target

    def purge(self):
        logging.info('Purging all previously trained models!')
        self.deactivate()
        self._models = []
        return self._models

    def model_type(self, y, regression):
        if not isinstance(regression, bool):
            if hasattr(self.model, '_estimator_type'):
                if self.model._estimator_type == 'classifier':
                    regression = False
                else:
                    regression = True
            elif isinstance(y[0], numbers.Number):
                regression = True
            else:
                regression = False
        return regression

    def model_train(self, X, Y, regression=None, max_time_mins=5 * 60, max_eval_time_mins=1, early_stop=5,
                    config_dict=TPOT_DEFAULT_CONFIG_DICT):

        params = dict(max_time_mins=max_time_mins,
                      max_eval_time_mins=max_eval_time_mins,
                      early_stop=early_stop,
                      random_state=RANDOM_STATE,
                      verbosity=3,
                      n_jobs=-1,
                      config_dict=config_dict)
        regression = self.model_type(Y, regression)
        if regression:
            model = Regressor(**params)
        else:
            model = Classifier(**params)
        model.fit(X, Y)
        model.warm_start = True
        return model

    def features_imputation(self, df, exclude=None, features=None, n_features=None, target=None):
        df, target = self.target_imputation(df, target)
        if features is None:
            if hasattr(self.model, 'n_features_in_'):
                n_features = getattr(self.model, 'n_features_in_')
            if isinstance(n_features, numbers.Number):
                if target in exclude:
                    exclude.remove(target)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if len(exclude) > 0:
                        correlation = df.drop(exclude, axis=1).corr(method='pearson')
                    else:
                        correlation = df.corr(method='pearson')
                    features = [c for c in correlation.nlargest(int(n_features) + 1, target).index
                        if c != target]
            else:
                features = self.valid_as_features_columns(df, target, exclude)
        elif not isinstance(features, Iterable):
            features = [features]
        return features, target

    def model_name_imputation(self, model_name, target):
        if model_name is None:
            if target not in self._models.keys():
                model_name = target
            else:
                for i in range(1000):
                    if f'{target}_{i}' not in self._models.keys():
                        model_name = f'{target}_{i}'
                        break
                else:
                    logging.error(' Please specify a model name to make!')
                    return
            try:
                model_name = model_name.replace(' ', '_')
            except AttributeError:
                logging.error(' Please specify a model name to make!')
                return
        return model_name

    def get_xy_from_df(self, df, features=None, target=None, n_features=None, regression=None, exclude=(),
                       drop_target_na=True, fill_features_na=True, trimXY=True, test_size=None, random_state=None,
                       train=True):
        if test_size is None:
            test_size = 0.2
        if random_state is None:
            random_state = RANDOM_STATE
        if features is None and self.active:
            features = self.features
        if target is None and self.active:
            target = self.target
        if n_features == 'all' or features == 'all':
            n_features = None
            features = None
        if drop_target_na:
            df = df.dropna(subset=[target])
        y = df[target].values
        regression = self.model_type(y, regression)
        if not regression:
            y = y > 0
        features, target = self.features_imputation(df, exclude=exclude, features=features, n_features=n_features,
                                                    target=target)
        if fill_features_na:
            x = df[features].fillna(method='bfill').values.astype('float32')
        else:
            x = df[features]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        if train:
            x, y = x_train, y_train
        else:
            x, y = x_test, y_test
        if trimXY:
            return self.trim_XY(x, y)
        else:
            return x, y

    def make_model(self, df, target, *args, model_name: str = None, features=None, n_features=10, regression=None,
                   exclude=(), test_size=None, random_state=None, **kwargs):
        self.deactivate()
        model_name = self.model_name_imputation(model_name, target)
        if model_name is None:
            return
        x, y = self.get_xy_from_df(df, features, target, n_features, regression, exclude,
                                   test_size=test_size, random_state=random_state, train=True)
        regression = self.model_type(y, regression)
        model_dict = self._models
        model_dict[model_name] = {'model': self.model_train(*self.trim_XY(x, y), *args, regression=regression,
                                                            **kwargs),
                                  'features': features,
                                  'target': target}
        self._models = model_dict
        self._last_model_name = model_name
        self.save()
        return model_dict

    def save(self, models: dict = None, filename=None):

        if models is None:
            models = self._models
        if filename is None:
            filename = self._filename

        if len(models) > 0:
            try:
                make_bak_file(filename)
                with open(filename, 'wb') as file:
                    pickle.dump(models, file)
                return True
            except Exception as e:
                logging.error(e)
        return False

    def load_models(self, filename=None, models: dict = None):
        if filename is None:
            filename = self.filename
        if models is None:
            try:
                with open(filename, 'rb') as file:
                    models = pickle.load(file)
            except (FileNotFoundError, EOFError, AttributeError) as e:
                logging.error(f' Unable to read models due to {e}.')
                return {}
        return models

    def get_method_and_model_name(self, model_name, models):
        if models is None:
            models = self._models.copy()
            fitted_model = (lambda model: model.fitted_pipeline_ if isinstance(model,
                            (Regressor, Classifier)) and hasattr(model,'fitted_pipeline_') else model)
            models = {k: {k1: v if k1 != 'model' else fitted_model(k1)} for k, md in models.items() for k1, v in
            md.items()}

        if model_name is None:
            model_name = self._last_model_name
        else:
            self._last_model_name = model_name
        return model_name, models

    def apply_mode(self, df, model_name=None, method_name=None,
                   trimXY=True, exclude=(), target=None, features=None, n_features=None,
                   **kwargs):
        x, y = np.array([]), np.array([])
        if method_name is None:
            logging.error(' Method name must be specified')
            return
        if self.model is None:
            if model_name is None:
                logging.error(' Model name must be specified.')
                return
            else:
                self._last_model_name = model_name
                if self.model is None:
                    logging.error(f' Model {model_name} not found.')
                    return
        if isinstance(df, pd.DataFrame):
            if features is None:
                features = self.features
            if features is None:
                features, target = self.features_imputation(df, exclude=exclude, n_features=n_features, target=target)
            if target is None:
                target = self.target
            if target is None:
                df, target = self.target_imputation(df, target)
            # noinspection PyUnresolvedReferences
            try:
                x = df[features].values
            except KeyError:
                x = np.array([])
            try:
                y = df[target].values
            except KeyError:
                y = np.array([])
        elif isinstance(df, pd.Series):
            if features is None:
                features = self.features
            if features is None:
                features, target = self.features_imputation(df, exclude=exclude, n_features=n_features, target=target)
            if target is None:
                target = self.target
            if target is None:
                df, target = self.target_imputation(df, target)
            try:
                x = df[features].values
            except KeyError:
                x = np.array([])
            try:
                y = np.array(df[target].values)
            except KeyError:
                y = np.array([])
        elif isinstance(df, Iterable):
            if hasattr(df, 'shape'):
                x = df
                if hasattr(model_name, 'shape'):
                    y = model_name
            elif len(df) == 2:
                if hasattr(df[0], 'shape'):
                    x, y = df
                elif len(df[0]) == len(features):
                    x = df[0]
                    y = df[1]
            elif len(df) == len(features) + 1:
                x = df[:-1]
                y = df[-1]
            else:
                if isinstance(df, numbers.Number):
                    x = [df]
                if isinstance(model_name, numbers.Number):
                    y = [model_name]
        if trimXY:
            x, y = self.trim_XY(x, y)
        if hasattr(self.model, method_name):
            kwargs['x'] = x
            kwargs['y'] = y
            return trim_run(getattr(self.model, method_name), kwargs)

    def activate(self, model_name: str = None):
        if model_name is None:
            logging.error(' Model name must be specified')
        elif model_name in self.models.keys():
            self._last_model_name = model_name
        else:
            logging.error(' Model name not found, please choose from existing models or train a new one.')

    def deactivate(self):
        self._last_model_name = None

    def test_features_relevance(self, df, model_name=None, model=None, features=None, normalize=True,
                                method=DEFAULT_IMP_METHOD, target=None, exclude=None, positive_only=True):
        x, y = [], []
        if model is None:
            if model_name is not None:
                self.activate(model_name)
            model = self.model
            if model is None:
                logging.error(' Please specify an existing model or train a new one.')
                return
        if target is None:
            target = self.target
        if features is None:
            features = self.features
            features, target = self.features_imputation(df, features=features, exclude=exclude, target=target)
        features_df = pd.DataFrame(columns=features)
        if method == 'all':
            return pd.DataFrame.from_dict({m: self.test_features_relevance(df, method=m) for m in
                                              ('coef', 'permutation', 'score', 'residual')})
        if method == 'coef':
            if hasattr(model, '_final_estimator'):
                estimator = model._final_estimator
            else:
                estimator = model
            if hasattr(estimator, 'coef_'):
                importance = estimator.coef_
            elif hasattr(estimator, 'feature_importances_'):
                importance = estimator.feature_importances_
            else:
                logging.error(
                    ' Final estimator for specified model has neither "coef_" nor "feature_importances_" attributes, '
                    'choose another test method.')
                return
            features_df = pd.Series(dict(zip(features, abs(importance)))).sort_values(ascending=False)
        else:
            x = df[features].applymap(partial(pd.to_numeric, errors='coerce')).values
            if method in ('permutation', 'score'):
                y = df[target].apply(partial(pd.to_numeric, errors='coerce')).values
                x, y = self.trim_XY(x, y)
            else:
                x = self.trim_XY(x)
            if method == 'permutation':
                if hasattr(model, '_final_estimator'):
                    estimator = model._final_estimator
                else:
                    estimator = model
                if hasattr(estimator, '_estimator_type') and estimator._estimator_type == 'regressor':
                    scoring = 'neg_mean_squared_error'
                else:
                    scoring = 'accuracy'
                try:
                    results = permutation_importance(model, x, y, scoring=scoring)
                    importance = results.importances_mean
                except ValueError:
                    importance = pd.Series([np.nan] * len(features))
                features_df = pd.Series(dict(zip(features, abs(importance))))
            elif method == 'residual':
                for row_index, row_feats in enumerate(tqdm(x)):
                    features_relevance = {}
                    baseline_prediction = model.predict([row_feats])
                    for index, feature_value in enumerate(row_feats):
                        new_row = row_feats.copy()
                        new_row[index] = -1 / feature_value if feature_value > 1e-6 else 1e6
                        new_prediction = model.predict([new_row])
                        features_relevance.update({index: abs(new_prediction - baseline_prediction)[0]})
                    features_df = features_df.append(dict(zip(features_df.columns, features_relevance.values())),
                                                     ignore_index=True)
                # noinspection PyArgumentList
                features_df = features_df.sum()
            elif method == 'score':
                baseline_score = model.score(x, y)
                for index, feature in enumerate(features_df.columns):
                    new_x = x.copy()
                    new_x[:, index] = x[:, index].mean()
                    features_df.loc[0, feature] = (baseline_score / model.score(new_x, y) - 1) * 100
                features_df = features_df.T[0]
            else:
                logging.error(' Valid methods are "coef", "permutation", "residual" and "score".')
                return

        if positive_only:
            features_df = features_df[features_df > 0]
        if normalize:
            min_max_scaler = MinMaxScaler()
            features_df = pd.Series((
                                            min_max_scaler.fit_transform(
                                                    (lambda values: values.reshape([len(values), 1]))
                                                    (features_df.values)) * 100).flatten(),
                                    features_df.keys()
                                    )
        features_df = features_df.sort_values(ascending=False)
        return features_df
