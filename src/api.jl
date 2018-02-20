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
# mutatable structure (a subtype `Koala.Model`) whose instantiations
# are the "models" which store the hyperparameters of the
# algorithm. An example of a hyperparameter is `min_patterns_split`;
# this is a field of `TreeRegressor` in `KoalaTrees` which controls
# the degree of pruning of decision trees.

# Fitting a model to training data (see below) produces a "predictor"
# which is not part of the model. However, the *type* of the predictor
# must be declared implicitly when defining the model type, by an
# appropriately subtyping as shown below. For example, the predictor
# type for the `ConstantRegressor` model type (for simply predicting
# the target mean for any input pattern) is `Float64`.

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

mutable struct SomeSupervisedModelType <: Regressor{CorrespondingPredictorType}

   ...hyperparameters declared here with invariants enforced with an inner constructor...

end

# or

mutable struct SomeSupervisedModelType <: Classifier{CorrespondingPredictorType}

    ...hyperparameters declared here with invariants enforced with an inner constructor...

end

# for brevity in this template (do a query/replace instead):
const M = SomeSupervisedModelType

# lazy keywork constructor
M(; param1=default1, parmam2=defalut2, etc) = M(param1, param2, etc)

# The output `scheme_X` of the following should include everything
# needed to transform the training or testing input data into the form
# required for the `fit` algorithm. For example, in a neural network,
# this might include standardization of the inputs. The transformation
# should furthermore only retain features with labels in
# `features`.
#
# To avoid data leakage, the transformation should depend only on
# training data, which this routine takes to be the data in
# `X[train_rows, features]`.
#
# Finally, one should include in `scheme_X` any "metadata" which
# `fit` may require for reporting purposes, such as the feature
# labels, `features`.
get_scheme_X(model::M, X::AbstractDataFrame, train_rows, features) -> scheme_X

# The actual transformation of inputs is carried out by following
# `transform` method, which should first check `X` is compatible with
# `scheme_X`; in particular, the feature labels of `X` should include
# all those encoded in scheme_X:
transform(model::M, scheme_X, X::AbstractDataFrame) -> Xt 

# Similar comments apply to transformations of the target, except that
# an `inverse_transform` method must also be provided:
get_scheme_y(model::M, y, train_rows) -> scheme_y
transform(model::M, scheme_y, y::AbstractVector{T} where T<:Real) -> yt
inverse_transform(model::M, scheme_y, yt) -> y

# Any preliminary part of training that does not depend on the value
# of `model` (ie it's field values) can be placed in `setup`. In
# iterative training algorithms, any such calculation which need not
# be repeated when iterations are added should go in `setup`. All
# other training should go in `fit`. All results of `setup` needed for
# the rest of training must be returned as `cache` for passing to
# `fit`.
setup(model::M, Xt, yt, scheme_X, parallel, verbosity) -> cache

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
fit(model::M, cache, add, parallel, verbosity; args...) -> predictor, report, cache

# The following uses `predictor` to predict *transformed* target
# values on new *transformed* input patterns. 
predict(model::M, predictor::P, Xt, parallel, verbosity) -> transformed_predictions


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





