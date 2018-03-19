# KoalaLow

> For exposing Koala's low-level interface.

Koala's high-level interface operates primarily on
*machines*. Presently, the Koala objects which can be wrapped in
machines are *supervised models* and *transformers*.  Koala's
low-level interface amounts to definitions of the following methods,
for each supervised model type `SMT <: Koala.SupervisedModel`, and
transformer type `TT <: Koala.Transformer`:

- `default_transformer_X(model::SMT)`

- `default_transformer_y(model::SMT)`

- `clean!(model::SMT)`

- `setup(model::SMT, Xt, yt, scheme_X, parallel, verbosity)`

- `fit(model::SMT, cache, add, parallel, verbosity; args...)`

- `predict(model::SMT, predictor, Xt, parallel, verbosity)`

- `fit(transformer::TT, X, parallel, verbosity; args...)`

- `transform(transformer::TT, scheme, X)`

- `inverse_transform(transformer::TT, scheme, Xt)` (optional)

The first two methods on the list have fall-backs defined in `Koala`. 

It is our intention that the low-level implementation of most
transformers appear in `KoalaTransforms` (a few are in `Koala`
itself), while each supervised model (or family of similar such
models) gets their own dedicated module (e.g., `KoalaTrees`).

The code,

````julia
using Koala
````

brings `default_transformer_X`, `default_transformer_y`, `clean!` and
`predict` into scope. To bring any other low-level method into scope, one imports it explicitly from `Koala`; to bring all of them into scope, use

````
using KoalaLow
````

For more on the low-level interface, see the template for implementing
new supervised learning algorithms at
[SupervisedModelTemplate.jl](src/SupervisedModelTemplate.jl), and the
template for new transformers at
[TransformerTemplate.jl](src/TransformerTemplate.jl).
