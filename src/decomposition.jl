struct Decomposition{I}
    #
    # We represent structured decompositions as diagrams D: I → Set:
    #
    #   ┌─────────────────────┐
    #   │    C     E          │
    #   │    ⇊     ⇊          │
    #   │    T  ←  R  →  V    │
    #   │       ↖  ↑          │
    #   │          S          │
    #   └─────────────────────┘
    #
    # where
    #
    #   - T is a set of nodes
    #   - V is a set of vertices
    #   - E is a set of edges
    #   - C is a set of tree edges
    #   - R is a set of residual incidences
    #   - S is a set of separator incidences
    #
    # The map R → V is a bijection, so R is a relabeling of V. Everything
    # else is expressed in terms of R.
    #
    # # Attributes
    #
    # The instance:
    #
    #    E ⇉ R
    #
    inst::Instance{I}
    #
    # The residuals:
    #
    #    T ← R → V
    #
    res::Span{I}
    #
    # The separators:
    #
    #    T ← S → R
    #
    sep::Span{I}
    #
    # The tree:
    #
    #    T ← C → T
    #
    tree::Span{I}
end

function gaifman(S::Span{I}) where {I}
    nv = S.nv
    T = invspan(S)

    marker = FVector{I}(undef, nv)

    for v in oneto(nv)
        marker[v] = zero(I)
    end

    ne = zero(I)

    for v in oneto(nv)
        tag = v
        pstrt = T.xadj[v]
        pstop = T.xadj[v + one(I)] - one(I)

        for p in pstrt:pstop
            u = T.adjncy[p]
            qstrt = S.xadj[u]
            qstop = S.xadj[u + one(I)] - one(I)

            for q in qstrt:qstop
                w = S.adjncy[q]

                if w != v && marker[w] < tag
                    marker[w] = tag
                    ne += one(I)
                end
            end
        end
    end

    xadj   = FVector{I}(undef, nv + one(I))
    adjncy = FVector{I}(undef, ne)

    r = one(I)

    for v in oneto(nv)
        tag = nv + v
        pstrt = T.xadj[v]
        pstop = T.xadj[v + one(I)] - one(I)
        xadj[v] = r

        for p in pstrt:pstop
            u = T.adjncy[p]
            qstrt = S.xadj[u]
            qstop = S.xadj[u + one(I)] - one(I)

            for q in qstrt:qstop
                w = S.adjncy[q]

                if w != v && marker[w] < tag
                    marker[w] = tag
                    adjncy[r] = w
                    r += one(I)
                end
            end
        end
    end

    xadj[nv + one(I)] = r
    return Span{I}(nv, nv, ne, xadj, adjncy)
end

function decomposition(X::Instance{I}, S::Span{I}; kw...) where {I}
    nu = X.dom.nu
    nv = S.nv
    #
    # relabel the elements in topological order of their strongly
    # connected components, sources first
    #
    C = sccs(X.dom)
    scc = FVector{I}(undef, nu)

    for u in oneto(nu)
        scc[u] = C.adjncy[nu + one(I) - u]
    end

    X = permute(X, scc)
    S = permute(S, scc)

    G = gaifman(S)
    graph = BipartiteGraph(G.nv, G.nu, G.ne, G.xadj, G.adjncy)
    perm, clique = cliquetree(graph; kw...)
    nt = convert(I, length(clique))

    L = invspan(permute(invspan(S), perm))

    node = FVector{I}(undef, nv)
    prnt = FVector{I}(undef, nt)
    fst  = FVector{I}(undef, nu)
    xres = FVector{I}(undef, nt + one(I))
    back = FVector{I}(undef, nu)

    for t in oneto(nt)
        for v in residual(clique, t)
            node[v] = t
        end

        p = parentindex(clique, t)

        if !isnothing(p)
            prnt[t] = p
        elseif t < nt
            prnt[t] = nt
        else
            prnt[t] = zero(I)
        end
    end

    for t in oneto(nt + one(I))
        xres[t] = zero(I)
    end

    for u in oneto(nu)
        pstrt = L.xadj[u]
        pstop = L.xadj[u + one(I)] - one(I)

        if pstrt <= pstop
            fst[u] = L.adjncy[pstrt]
            t = node[fst[u]]
        else
            fst[u] = nv + one(I)
            t = nt
        end

        xres[t + one(I)] += one(I)
    end

    xres[one(I)] = q = one(I)

    for t in oneto(nt)
        xres[t + one(I)] = q += xres[t + one(I)]
    end

    order = invmap(nv + one(I), nu, fst).adjncy
    #
    # The search assigns each group (elements with the same first
    # support vertex) from its end. Move one element with an arc into a
    # later group, which is already assigned, to the end: it is
    # enumerated from a preimage list, and the rest of the group follows
    # from it by images or is checked against it at once.
    #
    i = one(I)

    while i <= nu
        j = i

        while j < nu && fst[order[j + one(I)]] == fst[order[i]]
            j += one(I)
        end

        for p in j:-one(I):i
            u = order[p]
            found = false

            for e in X.dom.xadj[u]:X.dom.xadj[u + one(I)] - one(I)
                if fst[X.dom.adjncy[e]] > fst[u]
                    found = true
                    break
                end
            end

            if found
                for q in p:j - one(I)
                    order[q] = order[q + one(I)]
                end

                order[j] = u
                break
            end
        end

        i = j + one(I)
    end

    for u in oneto(nu)
        back[u] = scc[order[u]]
    end

    res = Span{I}(nt, nu, nu, xres, back)
    tree = invmap(nt, nt, prnt)
    Y = permute(X, order)
    R = permute(L, order)
    sep = separators(R, clique, nt)
    return Decomposition{I}(Y, res, sep, tree)
end

function separators(R::Span{I}, clique, nt::I) where {I}
    nu = R.nu
    nv = R.nv

    C = CliqueTrees.separators(clique)
    xfst = FVector{I}(undef, nv + one(I))

    for v in oneto(nv + one(I))
        xfst[v] = zero(I)
    end

    for u in oneto(nu)
        pstrt = R.xadj[u]
        pstop = R.xadj[u + one(I)] - one(I)

        if pstrt <= pstop
            xfst[R.adjncy[pstrt] + one(I)] += one(I)
        end
    end

    xfst[one(I)] = q = one(I)

    for v in oneto(nv)
        xfst[v + one(I)] = q += xfst[v + one(I)]
    end

    nempty = nu - xfst[nv + one(I)] + one(I)
    ne = zero(I)

    for t in oneto(nt)
        kstrt = pointers(C)[t]
        kstop = pointers(C)[t + one(I)] - one(I)

        for k in kstrt:kstop
            v = targets(C)[k]
            ustrt = xfst[v]
            ustop = xfst[v + one(I)] - one(I)

            for u in ustrt:ustop
                pstrt = R.xadj[u] + one(I)
                pstop = R.xadj[u + one(I)] - one(I)
                j = k + one(I)
                found = true

                for p in pstrt:pstop
                    w = R.adjncy[p]

                    while j <= kstop && targets(C)[j] < w
                        j += one(I)
                    end

                    if j > kstop || targets(C)[j] != w
                        found = false
                        break
                    end

                    j += one(I)
                end

                if found
                    ne += one(I)
                end
            end
        end

        if t < nt
            ne += nempty
        end
    end

    xadj   = FVector{I}(undef, nt + one(I))
    adjncy = FVector{I}(undef, ne)

    q = one(I)

    for t in oneto(nt)
        kstrt = pointers(C)[t]
        kstop = pointers(C)[t + one(I)] - one(I)
        xadj[t] = q

        for k in kstrt:kstop
            v = targets(C)[k]
            ustrt = xfst[v]
            ustop = xfst[v + one(I)] - one(I)

            for u in ustrt:ustop
                pstrt = R.xadj[u] + one(I)
                pstop = R.xadj[u + one(I)] - one(I)
                j = k + one(I)
                found = true

                for p in pstrt:pstop
                    w = R.adjncy[p]

                    while j <= kstop && targets(C)[j] < w
                        j += one(I)
                    end

                    if j > kstop || targets(C)[j] != w
                        found = false
                        break
                    end

                    j += one(I)
                end

                if found
                    adjncy[q] = u
                    q += one(I)
                end
            end
        end

        if t < nt
            ustrt = xfst[nv + one(I)]

            for u in ustrt:nu
                adjncy[q] = u
                q += one(I)
            end
        end
    end

    xadj[nt + one(I)] = q
    return Span{I}(nt, nu, ne, xadj, adjncy)
end

#
# Restrict the decomposition D to node t, writing the contraction into
# the given workspaces. Each element of the bag is stamped with its
# label in the bag (pos).
#
function restrict!(
        D::Decomposition{I},
        pos::AbstractVector{I},
        xadj::FVector{I},
        adjncy::FVector{I},
        vtx::FVector{I},
        arc::FVector{I},
        sxadj::FVector{I},
        sadjncy::FVector{I},
        t::I,
    ) where {I}
    X = D.inst
    rstrt = D.res.xadj[t]
    rstop = D.res.xadj[t + one(I)] - one(I)
    sstrt = D.sep.xadj[t]
    sstop = D.sep.xadj[t + one(I)] - one(I)
    cstrt = D.tree.xadj[t]
    cstop = D.tree.xadj[t + one(I)] - one(I)

    nr = rstop - rstrt + one(I)
    m = nr + sstop - sstrt + one(I)
    nc = cstop - cstrt + one(I)

    for x in oneto(nr)
        pos[rstrt + x - one(I)] = x
    end

    for x in nr + one(I):m
        pos[D.sep.adjncy[sstrt + x - nr - one(I)]] = x
    end
    #
    # the instance
    #
    xadj[one(I)] = ne = one(I)

    for x in oneto(m)
        if x <= nr
            g = rstrt + x - one(I)
        else
            g = D.sep.adjncy[sstrt + x - nr - one(I)]
        end

        vtx[x] = X.vtx[g]
        pstrt = X.dom.xadj[g]
        pstop = X.dom.xadj[g + one(I)] - one(I)

        for p in pstrt:pstop
            adjncy[ne] = pos[X.dom.adjncy[p]]
            arc[ne] = X.arc[p]
            ne += one(I)
        end

        xadj[x + one(I)] = ne
    end
    #
    # the children
    #
    sxadj[one(I)] = ns = one(I)

    for j in oneto(nc)
        c = D.tree.adjncy[cstrt + j - one(I)]
        pstrt = D.sep.xadj[c]
        pstop = D.sep.xadj[c + one(I)] - one(I)

        for p in pstrt:pstop
            sadjncy[ns] = pos[D.sep.adjncy[p]]
            ns += one(I)
        end

        sxadj[j + one(I)] = ns
    end

    dom = Span{I}(m, m, ne - one(I), xadj, adjncy)
    inst = Instance{I}(dom, X.cod, vtx, arc)
    sep = Span{I}(nc, m, ns - one(I), sxadj, sadjncy)
    return Contraction{I}(nr, inst, sep)
end

#
# The workspace sizes: the largest bag, the largest number of children
# of a node, the largest number of separator incidences of the children
# of a node, and the largest number of edges in a bag.
#
function bagsizes(D::Decomposition{I}) where {I}
    X = D.inst
    nt = D.tree.nu
    mbag = zero(I)
    mchd = zero(I)
    msep = zero(I)
    medg = zero(I)

    for t in oneto(nt)
        rstrt = D.res.xadj[t]
        rstop = D.res.xadj[t + one(I)] - one(I)
        sstrt = D.sep.xadj[t]
        sstop = D.sep.xadj[t + one(I)] - one(I)
        cstrt = D.tree.xadj[t]
        cstop = D.tree.xadj[t + one(I)] - one(I)
        n = zero(I)
        d = zero(I)

        for j in cstrt:cstop
            c = D.tree.adjncy[j]
            n += D.sep.xadj[c + one(I)] - D.sep.xadj[c]
        end

        for x in rstrt:rstop
            d += X.dom.xadj[x + one(I)] - X.dom.xadj[x]
        end

        for p in sstrt:sstop
            x = D.sep.adjncy[p]
            d += X.dom.xadj[x + one(I)] - X.dom.xadj[x]
        end

        mbag = max(mbag, rstop - rstrt + sstop - sstrt + two(I))
        mchd = max(mchd, cstop - cstrt + one(I))
        msep = max(msep, n)
        medg = max(medg, d)
    end

    return mbag, mchd, msep, medg
end
