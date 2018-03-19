# A module template for Koala supervised models

## INTRODUCTION

# This file briefly describes the low-level methods that must be
# implemented for each new Koala supervised learning algorithm. If
# this algorithm is called "SomeAlgorithm", then this implementation
# is conventionally contained in a module called `KoalaSomeAlgorithm`,
# eg, `KoalaTrees`.  The reader implementing a new algorithm in Koala
# may find the present file useful as a template.

# Normally regressor and classifier models with similar underlying
# algorithms should be placed in the same module.

# For each supervised learning algorithm one defines a supervised
# model type, namely a mutatable structure (a subtype of
# `Koala.SupervisedModel`) whose instances, hereafter called "models",
# store parameters controlling the outcome of training. A parameter
# controlling *how*, rather than *what*, the algorithm does is not a
# model parameter for present purposes. When a model parameter is
# changed during training, a machine wrapped around the model (which
# includes transformed versions of the learning data - see Koala.jl)
# does not need to be reconstructed. Parameters describing the
# transformations to be applied to input or target data (eg, whether
# to standardize target values) are *not* viewed as model parameters,
# as changing them generally necessitates wrapping the model in a new
# machine.

# An example of a model parameter is `min_patterns_split`; this is a
# field of `TreeRegressor` in `KoalaTrees` which controls the degree
# of pruning of decision trees. The number of trees in a random forest
# model is another model parameter, but a flag saying whether or not
# the trees should be trained in parallel is not a model
# parameter.

# Fitting a model to training data (see below) produces a "predictor"
# which is not part of the model. However, the *type* of the predictor
# must be declared implicitly when defining the model type, by an
# appropriately subtyping as shown below. For example, the predictor
# type for the `ConstantRegressor` model type (for simply predicting
# the target mean for any input pattern) is `Float64`.

# Some general guidelines:

# 1. Avoid giving models inner constructors, keeping in mind that
# these are *mutable* types. (Any desired invariants can and should be
# be enforced in the `clean!` method.)

# 2. With the exception of the `fit` method for transformers (see
# `TransformerTemplate.jl.`) avoid type and dimension checking in
# low-level methods. They should not be needed.


################################
# MODULE TEMPLATE STARTS BELOW #
################################

module KoalaSomeAlgorithm # eg KoalaRidge

# new:
export SomeSupervisedModelType # eg, RidgeRegressor

# needed in this module:
import Koala: Regressor, Classifier, softwarn
import DataFrames: AbstractDataFrame # the form of all untransformed input data

# to be extended (but not explicitly rexported):
import Koala: setup, fit, predict
import Koala: default_transformer_X, default_transformer_y, clean!
import Koala: transform, inverse_transform


## MODEL TYPE DEFINITIONS

mutable struct SomeSupervisedModelType <: Regressor{CorrespondingPredictorType}
    # eg ConstantRegressor <: Regressor{Float64}
    param1::Float64
    param2::Bool
    ...and so on...
end
# or
mutable struct SomeSupervisedModelType <: Classifier{CorrespondingPredictorType}
    param1::Float64
    param2::Bool
    ...and so on...
end

# lazy keywork constructor:
function SomeSupervisedModelType(; param1=default1, parmam2=defalut2, etc)
    model = SomeSupervisedModelType(param1, param2, etc)
    softwarn(clean!(model)) # only issues warning if `clean!` changes `model`
    return model
end

# The following optional method may change the fields of `model` to
# attempt to enforce natural invariants and avoid errors in later
# calls to `fit`. Any changes should be reported to the returned
# string (empty if the model parameters are fine). As a last resort, it
# may throw an exception. The high-level `fit!` method always calls
# this method on the model before doing anything else.
function clean!(model::SomeSupervisedModelType) -> message


## TRANSFORMERS

# For each model default `Transformer` objects must be specified with
# the following methods. The transformers are used to transform the
# input and output patterns into a form acceptable to the model's
# corresponding `fit` method. If the fall-backs
# `transformer_X=FeatureTruncater()` and
# `transformer_y=IdentityTransformer()` given in `Koala` suffice, then
# omit the two lines below; otherwise ammend appropriately:
default_transformer_X(model::SomeSupervisedModelType) -> transformer_X::Transformer
default_transformer_y(model::SomeSupervisedModelType) -> transformer_y::Transformer

# Notes:

# 1. Of course if `transformer_X` and `transformer_y` are of new
#   transformer types, then corresponding `fit` and `transform`
#   methods will need to be provided (see TransformerTemplate.jl). An
#   `inverse_transform` method is required only in the case of
#   `transformer_y`. At the very least, the scheme that `transform`
#   outputs should record the labels of the features of the dataframe
#   `X` given as input, so that only these are retained in calls to
#   `transform` on test data.

# 2. Any other "metadata" needed for reporting, in calls to the
#   supervised model's `fit` method defined below, should also be
#   stored in the scheme, `scheme_X`, output by `fit(transformer_X,
#   etc...)`.  The metadata should be extracted from `scheme_X` in
#   `setup` below and bundled in the `cache` part of that method's
#   return value. In this way `fit` will have access to the metadata
#   through its `cache` input.

# 3. To avoid data leakage, it is understood that transformers are
#   only fit on training data.


## TRAINING AND PREDICTION METHODS

# Any preliminary part of training that does *not* depend on the value
# of `model` (ie it's field values) can be placed in `setup`. (It is
# assumed that the data received by `setup` is already transformed.)
# Note that every time new data is given to the high-level method
# `fit!`, `setup` will be called anew.  In iterative training
# algorithms, any such calculation which needn't be repeated when
# iterations are added should go in `setup`. All other training should
# go in `fit`. All results of `setup` needed for the rest of training
# must be returned as `cache` for passing to `fit`.
setup(model::SomeSupervisedModelType, Xt, yt,
      scheme_X, parallel, verbosity) -> cache

# The value of `predictor` returned below need only include the
# minimum data needed to furnish predictions on new patterns. In
# iterative training algorithms, data that must be saved for further
# iterations (eg, `Xt`, `yt`) should be stored and returned in
# `cache`. Store in `report` post-fit statistics, such as feature
# rankings or training (in sample) error.
#
# Calling `fit` with `add=true` for an iterative training algorithm
# should add iterations.
#
# Care should be taken that successive calls to `fit` with `add=false`
# return *distinct* `predictor` objects. We allow `fit` to change the
# value of `cache` (but do not use the name "fit!" to distiguish the
# method from the high-level method with that name). You may design
# `fit` to take keyword arguments, `args` (to implement special
# training instructions) and if the high-level method `fit!` is called
# with these arguments on a corresponding `SupervisedMachine` object,
# then these will be passed along to the low-level `fit` method below.
function fit(model::SomeSupervisedModelType, cache, add, parallel, verbosity; args...)
    return predictor, report, cache
end
    
# The following uses `predictor` to predict *transformed* target
# values on new *transformed* input patterns. 
predict(model::SomeSupervisedModelType,
        predictor::CorrespondingPredictorType,
        Xt, parallel, verbosity) -> transformed_predictions

# Note that `CorrespondingPredictorType` must be the type appearing in
# the type declaration of `SomeSupervisedModelType` above.


end # of module





