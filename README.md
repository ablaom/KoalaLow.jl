# KoalaLow

> For exposing Koala's low-level interface.

Koala's low-level API consists of methods for the following
functions: 

- `default_transformer_X`
- `default_transformer_y`
- `setup`
- `fit`
- `predict`
- `transform`
- `inverse_transform`

Of these, only `default_transformer_X`, `default_transformer_y` and
`predict` are brought into scope by running `using Koala` (the last, because
it is overloaded for use on `SupervisedMachine` objects). To bring the
other methods explicitly into scope do this:

````julia
    julia> using KoalaLow
````

For more on the low-level interface, see the template for implementing
(or wrapping) new supervised learning algorithms at
[SupervisedModelTemplate.jl](src/SupervisedModelTemplate.jl), and the
template for new transformers at
[TransformerTemplate.jl](src/TransformerTemplate.jl).
