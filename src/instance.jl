struct Instance{I}
    #
    # We represent instances as diagrams D: I → Set:
    #
    #   ┌─────────────────┐
    #   │     E  ⇉  V     │
    #   │ arc ↓     ↓ vtx │
    #   │     F  ⇉  W     │
    #   └─────────────────┘
    #
    # where
    #
    #   - V is a set of vertices
    #   - E is a set of edges
    #   - W is a set of vertex types
    #   - F is a set of edge types
    #
    # # Attributes
    #
    # The graph of vertices and edges:
    #
    #    E ⇉ V
    #
    dom::Span{I}
    #
    # The graph of vertex types and edge types:
    #
    #    F ⇉ W
    #
    cod::Span{I}
    #
    # Each vertex v ∈ V has a type
    #
    #    vtx[v] ∈ W
    #
    vtx::FVector{I}
    #
    # Each edge e ∈ E has a type
    #
    #    arc[e] ∈ F
    #
    arc::FVector{I}
end

function invvtx(X::Instance{I}) where {I}
    return invmap(X.cod.nu, X.dom.nu, X.vtx)
end

function invarc(X::Instance{I}) where {I}
    return invmap(X.cod.ne, X.dom.ne, X.arc)
end

function permute(X::Instance{I}, order::AbstractVector{I}) where {I}
    nv = X.dom.nu
    ne = X.dom.ne

    index  = FVector{I}(undef, nv)
    vtx    = FVector{I}(undef, nv)
    arc    = FVector{I}(undef, ne)
    xadj   = FVector{I}(undef, nv + one(I))
    adjncy = FVector{I}(undef, ne)

    for v in oneto(nv)
        index[order[v]] = v
    end

    q = one(I)

    for v in oneto(nv)
        w = order[v]
        pstrt = X.dom.xadj[w]
        pstop = X.dom.xadj[w + one(I)] - one(I)
        vtx[v] = X.vtx[w]
        xadj[v] = q

        for p in pstrt:pstop
            adjncy[q] = index[X.dom.adjncy[p]]
            arc[q] = X.arc[p]
            q += one(I)
        end
    end

    xadj[nv + one(I)] = q

    dom = Span{I}(nv, nv, ne, xadj, adjncy)
    return Instance{I}(dom, X.cod, vtx, arc)
end

function saturate(X::Instance{I}, seed::Span{I}) where {I}
    nu = seed.nu
    nv = seed.nv

    umark = FVector{I}(undef, nu)
    vmark = FVector{I}(undef, nv)
    stack = FVector{I}(undef, nu)

    for u in oneto(nu)
        umark[u] = zero(I)
    end

    for v in oneto(nv)
        vmark[v] = zero(I)
    end

    ne = zero(I)

    for u in oneto(nu)
        tag = u
        umark[u] = tag
        stack[one(I)] = u
        n = one(I)

        while ispositive(n)
            w = stack[n]
            n -= one(I)
            pstrt = seed.xadj[w]
            pstop = seed.xadj[w + one(I)] - one(I)

            for p in pstrt:pstop
                v = seed.adjncy[p]

                if vmark[v] < tag
                    vmark[v] = tag
                    ne += one(I)
                end
            end

            estrt = X.dom.xadj[w]
            estop = X.dom.xadj[w + one(I)] - one(I)

            for e in estrt:estop
                x = X.dom.adjncy[e]

                if umark[x] < tag
                    umark[x] = tag
                    n += one(I)
                    stack[n] = x
                end
            end
        end
    end

    xadj   = FVector{I}(undef, nu + one(I))
    adjncy = FVector{I}(undef, ne)

    q = one(I)

    for u in oneto(nu)
        tag = nu + u
        umark[u] = tag
        xadj[u] = q
        stack[one(I)] = u
        n = one(I)

        while ispositive(n)
            w = stack[n]
            n -= one(I)
            pstrt = seed.xadj[w]
            pstop = seed.xadj[w + one(I)] - one(I)

            for p in pstrt:pstop
                v = seed.adjncy[p]

                if vmark[v] < tag
                    vmark[v] = tag
                    adjncy[q] = v
                    q += one(I)
                end
            end

            estrt = X.dom.xadj[w]
            estop = X.dom.xadj[w + one(I)] - one(I)

            for e in estrt:estop
                x = X.dom.adjncy[e]

                if umark[x] < tag
                    umark[x] = tag
                    n += one(I)
                    stack[n] = x
                end
            end
        end
    end

    xadj[nu + one(I)] = q
    return Span{I}(nu, nv, ne, xadj, adjncy)
end

#
# The image of y under arc a, for Y normalized.
#
function image(Y::Instance{I}, y::I, a::I) where {I}
    return Y.dom.adjncy[Y.dom.xadj[y] + a - Y.cod.xadj[Y.vtx[y]]]
end
