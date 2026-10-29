#
# The weights W are either nothing or a vector with one entry per sort
# c: either nothing (every image has weight 1) or a matrix of size
# |Y(c)| x |X(c)| whose entry (i, j) is the weight of mapping the j-th
# element of X of sort c to the i-th element of Y of sort c, both in
# the caller's numbering. A weight of 0 forbids the image.
#
# The weight of a homomorphism is the product of the weights of its
# images, and score computes the sum of the weights of all
# homomorphisms X -> Y.
#
# The strategy is Eager or Lazy. The eager search computes, at each
# node, the table of every separator assignment that extends into the
# subtree, children first. The lazy search asks each child only about
# the separator assignments its parent reaches, and stops at the first
# homomorphism. The lazy search requires the Boolean semiring.
#
@enum Strategy Eager Lazy

#
# The default strategy: Lazy for the Boolean semiring, and Eager
# otherwise.
#
function default_strategy(R::AbstractSemiring)
    if R isa Boolean
        return Lazy
    else
        return Eager
    end
end

function score(R::AbstractSemiring, W, X::Instance{I}, S::Span{I}, Y::Instance{I}; strategy::Strategy = default_strategy(R), kw...) where {I}
    D = decomposition(X, S; kw...)
    return score(R, W, D, Y; strategy)
end

function score(R::AbstractSemiring, W, D::Decomposition{I}, Y::Instance{I}; strategy::Strategy = default_strategy(R)) where {I}
    value, h = solve(R, W, D, Y, strategy, false)
    return value
end

#
# Find the score and a homomorphism attaining it, or nothing if the
# score is 0. The semiring must be selective.
#
function color(R::AbstractSemiring, W, X::Instance{I}, S::Span{I}, Y::Instance{I}; strategy::Strategy = default_strategy(R), kw...) where {I}
    D = decomposition(X, S; kw...)
    return color(R, W, D, Y; strategy)
end

function color(R::AbstractSemiring, W, D::Decomposition{I}, Y::Instance{I}; strategy::Strategy = default_strategy(R)) where {I}
    @assert isselective(R)
    return solve(R, W, D, Y, strategy, true)
end

#
# Find a homomorphism X -> Y, or nothing.
#
function color(W, X::Instance{I}, S::Span{I}, Y::Instance{I}; kw...) where {I}
    value, h = color(Boolean(), W, X, S, Y; kw...)
    return h
end

function color(X::Instance{I}, S::Span{I}, Y::Instance{I}; kw...) where {I}
    return color(nothing, X, S, Y; kw...)
end

function color(W, D::Decomposition{I}, Y::Instance{I}; kw...) where {I}
    value, h = color(Boolean(), W, D, Y; kw...)
    return h
end

function color(D::Decomposition{I}, Y::Instance{I}; kw...) where {I}
    return color(nothing, D, Y; kw...)
end

#
# The type of the weights: the element type of the first weight
# matrix, or the default for the semiring.
#
function valtype(R::AbstractSemiring, W)
    if !isnothing(W)
        for M in W
            isnothing(M) || return eltype(M)
        end
    end

    return valtype(R)
end

#
# Relabel Y and the weights, and search with the strategy. Unless
# witness is false, also find a homomorphism attaining the score.
#
function solve(R::AbstractSemiring, W, D::Decomposition{I}, Y::Instance{I}, strategy::Strategy, witness::Bool) where {I}
    Y, yord = normalize(Y)
    typ = invvtx(Y)
    base, pre = preimages(Y, typ)
    xwgt, wgt = weights(W, D, typ, valtype(R, W))

    if strategy == Lazy
        if !(R isa Boolean)
            throw(ArgumentError("the lazy strategy requires the Boolean semiring"))
        end

        return lazy(R, D, Y, yord, pre, base, typ, xwgt, wgt, witness)
    end
    #
    # the factor (back-pointers) is kept only for a witness
    #
    if witness
        fac = Vector{I}(undef, 1024)
    else
        fac = nothing
    end

    return eager(R, D, Y, yord, pre, base, typ, xwgt, wgt, fac)
end

#
# The preimages of the elements of Y under each arc. The preimages of y
# under arc a are the column
#
#    base[a] + y - typ.xadj[tgt(a)]
#
# of the returned span, listed in increasing order.
#
function preimages(Y::Instance{I}, typ::Span{I}) where {I}
    ny = Y.dom.nu
    nf = Y.dom.ne
    na = Y.cod.ne

    base = FVector{I}(undef, na + one(I))
    base[one(I)] = one(I)

    for a in oneto(na)
        d = Y.cod.adjncy[a]
        base[a + one(I)] = base[a] + typ.xadj[d + one(I)] - typ.xadj[d]
    end

    np = base[na + one(I)] - one(I)
    xadj   = FVector{I}(undef, np + one(I))
    adjncy = FVector{I}(undef, nf)

    for l in oneto(np)
        xadj[l + one(I)] = zero(I)
    end

    for y in oneto(ny)
        pstrt = Y.dom.xadj[y]
        pstop = Y.dom.xadj[y + one(I)] - one(I)

        for p in pstrt:pstop
            a = Y.arc[p]
            l = base[a] + Y.dom.adjncy[p] - typ.xadj[Y.cod.adjncy[a]]

            if l < np
                xadj[l + two(I)] += one(I)
            end
        end
    end

    xadj[one(I)] = q = one(I)

    for l in oneto(np)
        xadj[l + one(I)] = q += xadj[l + one(I)]
    end

    for y in oneto(ny)
        pstrt = Y.dom.xadj[y]
        pstop = Y.dom.xadj[y + one(I)] - one(I)

        for p in pstrt:pstop
            a = Y.arc[p]
            l = base[a] + Y.dom.adjncy[p] - typ.xadj[Y.cod.adjncy[a]]
            adjncy[xadj[l + one(I)]] = y
            xadj[l + one(I)] += one(I)
        end
    end

    return base, Span{I}(np, ny, nf, xadj, adjncy)
end

#
# Relabel the weights: element r of X gets the column
#
#    wgt[xwgt[r]] ... wgt[xwgt[r + 1] - 1]
#
# indexed by the elements of Y of its sort, or an empty column if every
# image has weight 1. Returns nothing, nothing if there are no weights.
#
function weights(W, D::Decomposition{I}, typ::Span{I}, ::Type{T}) where {I, T}
    X = D.inst
    nu = X.dom.nu
    ns = X.cod.nu
    nw = zero(I)

    if !isnothing(W)
        for r in oneto(nu)
            c = X.vtx[r]

            if !isnothing(W[c])
                nw += typ.xadj[c + one(I)] - typ.xadj[c]
            end
        end
    end

    if iszero(nw)
        return nothing, nothing
    end

    rank = FVector{I}(undef, nu)
    nsrt = FVector{I}(undef, ns)
    xwgt = FVector{I}(undef, nu + one(I))
    wgt = FVector{T}(undef, nw)
    #
    # rank[v] is first the sort of element v in the caller's numbering,
    # then its rank among the elements of that sort
    #
    for r in oneto(nu)
        rank[D.res.adjncy[r]] = X.vtx[r]
    end

    for c in oneto(ns)
        nsrt[c] = zero(I)
    end

    for v in oneto(nu)
        c = rank[v]
        nsrt[c] += one(I)
        rank[v] = nsrt[c]
    end

    xwgt[one(I)] = one(I)

    for r in oneto(nu)
        c = X.vtx[r]

        if isnothing(W[c])
            xwgt[r + one(I)] = xwgt[r]
        else
            xwgt[r + one(I)] = xwgt[r] + typ.xadj[c + one(I)] - typ.xadj[c]
        end
    end

    for r in oneto(nu)
        c = X.vtx[r]
        l = xwgt[r]

        if l < xwgt[r + one(I)]
            M = W[c]
            j = rank[D.res.adjncy[r]]

            for i in oneto(xwgt[r + one(I)] - l)
                wgt[l + i - one(I)] = M[i, j]
            end
        end
    end

    return xwgt, wgt
end

#
# Relabel Y so that its elements are grouped by type and each
# element's edges are listed in schema order. Returns the relabeled
# instance and the permutation from new labels to old.
#
function normalize(Y::Instance{I}) where {I}
    ny = Y.dom.nu
    nf = Y.dom.ne

    order = invvtx(Y).adjncy
    Z = permute(Y, order)

    arc    = FVector{I}(undef, nf)
    adjncy = FVector{I}(undef, nf)

    for y in oneto(ny)
        c = Z.vtx[y]
        pstrt = Z.dom.xadj[y]
        pstop = Z.dom.xadj[y + one(I)] - one(I)

        for p in pstrt:pstop
            a = Z.arc[p]
            q = pstrt + a - Z.cod.xadj[c]
            arc[q] = a
            adjncy[q] = Z.dom.adjncy[p]
        end
    end

    dom = Span{I}(ny, ny, nf, Z.dom.xadj, adjncy)
    return Instance{I}(dom, Z.cod, Z.vtx, arc), order
end
