# This file describes the methods that should be implemented for each
# new Koala transformer type. See https://github.com/ablaom/KoalaTransforms.jl
# for examples.

struct SomeTransformerType <: Transformer
    param1::Float64
    param2::Int
    etc...
end

# lazy keyword constructor:
SomeTransformerType(; param1=default1, param2=default2, etc...) =
    SomeTransformerType(param1, param2, etc...)

fit(transformer::SomeTransformerType, X, parallel, verbosity;
    optional_keyword_args...) -> scheme

transform(transformer::SomeTransformerType, scheme, X) -> Xt

# optional:
inverse_transform(transformer::SomeTransformerType, scheme, Xt) -> Xt


