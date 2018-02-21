# The Koala low-level API (and template)


## Introduction

# This file breifly describes the low-level methods that must be
# implemented for each new Koala learning algorithm. If this algorithm
# is called "SomeAlgorithm", then this implementation is
# conventionally contained in a module called `KoalaSomeAlgorithm`,
# eg, `KoalaTrees`.  Currently only supervised learning algorithms are
# supported. The reader implementing a new algorithm in Koala may find
# the present file useful as a template.

# For each learning algorithm one defines a "model type", namely a
# mutatable structure (a subtype `Koala.Model`) whose instances are
# the "models" storing the hyperparameters of the algorithm. An
# example of a hyperparameter is `min_patterns_split`; this is a field
# of `TreeRegressor` in `KoalaTrees` which controls the degree of
# pruning of decision trees. Generally, parameters whose values effect the
# outcome of model training are hyperparameters. So, the number of
# trees in a random forest model is another parameter, but a flag
# saying whether the trees should be trained in parallel is
# not. Parameters describing the transformations to be applied to
# input or target data (eg, whether to standardize target values) are
# deemed to be hyperparameters.

# Fitting a model to training data (see below) produces a "predictor"
# which is not part of the "model" in our sense. However, the *type*
# of the predictor must be declared implicitly when defining the model
# type, by an appropriately subtyping as shown below. For example, the
# predictor type for the `ConstantRegressor` model type (for simply
# predicting the target mean for any input pattern) is `Float64`.

################################
# Module template starts below #
################################

module SomeAlgorithm

# new:
export SomeSupervisedModelType

# needed in this module:
import Koala: Regressor
import DataFrames: AbstractDataFrame

# to be extended (but not explicitly rexported):
import Koala: setup, fit, predict
import Koala: get_scheme_X, get_scheme_y, transform, inverse_transform


## Model type definitions:

# The following should include inner constructors if invariants need to be enforced:
mutable struct SomeSupervisedModelType <: Regressor{CorrespondingPredictorType}
    hyperparam1::Float64
    hyperparam2::Bool
    ...and so on...
end
# or
mutable struct SomeSupervisedModelType <: Classifier{CorrespondingPredictorType}
    hyperparam1::Float64
    hyperparam2::Bool
    ...and so on...
end

# lazy keywork constructor
SomeSupervisedModelType(; param1=default1, parmam2=defalut2, etc) =
    SomeSupervisedModelType(param1, param2, etc)

# The output `scheme_X` of the following should include everything
# needed to transform the training or testing input data into the form
# required for the `fit` algorithm. For example, in a neural network,
# this might include standardization of the inputs. The transformation
# should furthermore only retain features with labels in the input
# vector `features`.
#
# To avoid data leakage, the transformation should depend only on
# training data, which this routine takes to be the data in
# `X[train_rows, features]`.
#
# Finally, one should include in `scheme_X` any "metadata" which `fit`
# may require for reporting purposes, such as the actual feature labels
# (after, say, one-hot encoding the categoricals).
get_scheme_X(model::SomeSupervisedModelType, X::AbstractDataFrame,
             train_rows, features) -> scheme_X

# The actual transformation of inputs is carried out by following
# `transform` method, which should first check `X` is compatible with
# `scheme_X`; in particular, the feature labels of `X` should include
# all those passed to `get_scheme_X` above:
transform(model::SomeSupervisedModelType, scheme_X, X::AbstractDataFrame) -> Xt 

# Similar comments apply to transformations of the target, except that
# an `inverse_transform` method must also be provided:
get_scheme_y(model::SomeSupervisedModelType, y, train_rows) -> scheme_y
transform(model::SomeSupervisedModelType, scheme_y,
          y::AbstractVector{T}) where T<:Real) -> yt
inverse_transform(model::SomeSupervisedModelType, scheme_y, yt) -> y

# Any preliminary part of training that does not depend on the value
# of `model` (ie it's field values) can be placed in `setup`. In
# iterative training algorithms, any such calculation which need not
# be repeated when iterations are added should go in `setup`. All
# other training should go in `fit`. All results of `setup` needed for
# the rest of training must be returned as `cache` for passing to
# `fit`.
setup(model::SomeSupervisedModelType, Xt, yt, scheme_X, parallel, verbosity) -> cache

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
# Care should be taken that successive calls to `fit` with
# `add=false` return *distinct* `predictor` objects. Although we use
# "fit" and not "fit!" here, we allow `fit` to change the value of
# `cache`. You may design `fit` to take keyword arguments, `args` (to
# implement special training instructions) and if `fit!` is called
# with these arguments on a corresponding `SupervisedMachine` object,
# then these will be passed along to `fit`.
function fit(model::SomeSupervisedModelType,
    cache, add, parallel, verbosity; args...)
    return predictor, report, cache
end
    
# The following uses `predictor` to predict *transformed* target
# values on new *transformed* input patterns. 
predict(model::SomeSupervisedModelType, predictor::P, Xt, parallel, verbosity) -> transformed_predictions


## Provided automatically by `Koala` (this is FYI and should be deleted):

# extensions :
setup(model::SupervisedModel, Xt, yt, train_rows, scheme_X, parallel, verbosity) =
    setup(model, Xt[train_rows,:], yt[train_rows], scheme_X, parallel, verbosity) 
predict(model::SupervisedModel, predictor, Xt, test_rows, parallel, verbosity) =
    predict(model, predictor, Xt[test_rows,:], parallel, verbosity)

# for low-level API error computations:
err(model::Regressor, predictor, X, y, rows, parallel, verbosity, loss::Function=rms) =
    loss(y[test_rows], predict(model, predictor, X, test_rows, parallel, verbosity))
err(model::Regressor, predictor, X, y, parallel, verbosity, loss::Function=rms) =
    return loss(y, predict(model, predictor, X, parallel, verbosity))

end # of module





