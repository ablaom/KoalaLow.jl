# KoalaLow

> For exposing Koala's low-level interface.

Koala's low-level API consists of methods for the following
functions: 

- `setup`
- `fit`
- `predict`
- `get_transformer_X`
- `get_transformer_y`
- `transform`
- `inverse_transform`

Of these, only `predict` is brought into scope by running `using
Koala` (because `SupervisedMachine` objects need it). To bring the
other methods explicitly into scope do this:

````julia
    julia> using KoalaLow
````

For more on the low-level interface, see the template for implementing
(or wrapping) new supervised learning algorithms at
[SupervisedModelTemplate.jl](src/SupervisedModelTemplate.jl), and the
template for new transformers at
[TransformerTemplate.jl](src/TransformerTemplate.jl).
