#
# Allocate the workspaces and run the forward pass and, for a witness,
# the backward pass. The weights xwgt, wgt and the factor fac are
# either vectors or nothing, and this function is compiled separately
# for each.
#
# The forward pass visits the nodes in postorder. At each node, the
# decomposition is restricted to a contraction, the arguments (the
# children's tables, in place, and the weights of the bag, copied) are
# gathered, and the local problem is solved by contract!. The tables
# live on a stack: the children's tables are on top, the table of the
# node is written above them and then moved down over them. For a
# witness, contract! also writes back-pointers to the factor.
#
# The backward pass visits the nodes from the root. At each node, the
# separator is assigned by the ancestors, and lift! reads the
# back-pointers to assign the residual and choose a column for each
# child.
#
function eager(
        R::AbstractSemiring,
        D::Decomposition{I},
        Y::Instance{I},
        yord::AbstractVector{I},
        pre::Span{I},
        base::AbstractVector{I},
        typ::Span{I},
        xwgt,
        wgt,
        fac,
    ) where {I}
    if isnothing(wgt)
        T = valtype(R)
    else
        T = eltype(wgt)
    end

    nu = D.inst.dom.nu
    nt = D.tree.nu
    mbag, mchd, msep, medg = bagsizes(D)
    #
    # the contraction
    #
    pos     = FVector{I}(undef, nu)
    xadj    = FVector{I}(undef, mbag + one(I))
    adjncy  = FVector{I}(undef, medg)
    vtx     = FVector{I}(undef, mbag)
    arc     = FVector{I}(undef, medg)
    sxadj   = FVector{I}(undef, mchd + one(I))
    sadjncy = FVector{I}(undef, msep)
    #
    # the arguments
    #
    ckey = FVector{I}(undef, mchd)
    cval = FVector{I}(undef, mchd)
    ccol = FVector{I}(undef, mchd)

    if isnothing(wgt)
        lxwgt = lwgt = nothing
    else
        lxwgt = FVector{I}(undef, mbag + one(I))
        lwgt  = Vector{T}(undef, 1024)
    end
    #
    # the local problem
    #
    work = ContractionWorkspace{I}(mbag, mchd, msep, medg)
    #
    # the backward step
    #
    hloc = FVector{I}(undef, mbag)
    sel  = FVector{I}(undef, mchd)
    #
    # the stack of tables: the table of node t has ncol[t] columns,
    # with keys tab[xtab[t]] ... and values tval[vtab[t]] ...
    #
    xtab = FVector{I}(undef, nt)
    vtab = FVector{I}(undef, nt)
    ncol = FVector{I}(undef, nt)
    tab  = Vector{I}(undef, 1024)
    tval = Vector{T}(undef, 1024)
    ntab = zero(I)
    nval = zero(I)
    #
    # the factor: the back-pointers of node t follow fac[xfac[t]]
    #
    if isnothing(fac)
        xfac = nothing
    else
        xfac = FVector{I}(undef, nt)
    end

    nfac = zero(I)

    for t in oneto(nt)
        C = restrict!(D, pos, xadj, adjncy, vtx, arc, sxadj, sadjncy, t)
        lwgt = gather!(D, xtab, vtab, ncol, xwgt, wgt, ckey, cval, ccol, lxwgt, lwgt, t)
        n, nf = contract!(R, C, Y, pre, base, typ, lxwgt, lwgt, ckey, cval, ccol, tab, tval, tab, tval, ntab, nval, fac, nfac, work)

        if iszero(n)
            return szero(R, T), nothing
        end
        #
        # pop the children's tables and move the table down over them
        #
        s = C.inst.dom.nu - C.nr
        kb = ntab
        vb = nval

        if ispositive(C.sep.nu)
            c = D.tree.adjncy[D.tree.xadj[t]]
            kb = xtab[c] - one(I)
            vb = vtab[c] - one(I)
        end

        for i in oneto(n * s)
            tab[kb + i] = tab[ntab + i]
        end

        if !(R isa Boolean)
            for i in oneto(n)
                tval[vb + i] = tval[nval + i]
            end
        end

        xtab[t] = kb + one(I)
        vtab[t] = vb + one(I)
        ncol[t] = n
        ntab = kb + n * s

        if !(R isa Boolean)
            nval = vb + n
        end

        if !isnothing(fac)
            xfac[t] = nfac
            nfac += nf
        end
    end
    #
    # in the Boolean semiring every stored entry has value 1, so values
    # are not stored
    #
    if R isa Boolean
        value = sone(R, T)
    else
        value = tval[vtab[nt]]
    end

    if isnothing(fac)
        return value, nothing
    end
    #
    # the backward pass: h is the assignment, and ncol[t] becomes the
    # choice for node t made by its parent
    #
    h = FVector{I}(undef, nu)
    ncol[nt] = one(I)

    for t in nt:-one(I):one(I)
        C = restrict!(D, pos, xadj, adjncy, vtx, arc, sxadj, sadjncy, t)
        nr = C.nr
        rstrt = D.res.xadj[t]
        sstrt = D.sep.xadj[t]
        cstrt = D.tree.xadj[t]
        #
        # the separator is assigned by the ancestors
        #
        for x in nr + one(I):C.inst.dom.nu
            hloc[x] = h[D.sep.adjncy[sstrt + x - nr - one(I)]]
        end

        lift!(C, fac, xfac[t], hloc, ncol[t], sel)

        for x in oneto(nr)
            h[rstrt + x - one(I)] = hloc[x]
        end

        for j in oneto(C.sep.nu)
            ncol[D.tree.adjncy[cstrt + j - one(I)]] = sel[j]
        end
    end

    out = FVector{I}(undef, nu)

    for r in oneto(nu)
        out[D.res.adjncy[r]] = yord[h[r]]
    end

    return value, out
end

#
# The arguments of the local problem at node t: the table of each
# child j, with ccol[j] columns, keys tab[ckey[j]] ... and values
# tval[cval[j]] ..., read in place; and the weights of each element x
# of the bag
#
#    lwgt[lxwgt[x]] ... lwgt[lxwgt[x + 1] - 1],
#
# copied. Returns the weights, which grow geometrically.
#
function gather!(
        D::Decomposition{I},
        xtab::AbstractVector{I},
        vtab::AbstractVector{I},
        ncol::AbstractVector{I},
        xwgt,
        wgt,
        ckey::AbstractVector{I},
        cval::AbstractVector{I},
        ccol::AbstractVector{I},
        lxwgt,
        lwgt,
        t::I,
    ) where {I}
    cstrt = D.tree.xadj[t]
    cstop = D.tree.xadj[t + one(I)] - one(I)
    #
    # the tables
    #
    for j in oneto(cstop - cstrt + one(I))
        c = D.tree.adjncy[cstrt + j - one(I)]
        ckey[j] = xtab[c]
        cval[j] = vtab[c]
        ccol[j] = ncol[c]
    end
    #
    # the weights
    #
    if !isnothing(wgt)
        rstrt = D.res.xadj[t]
        rstop = D.res.xadj[t + one(I)] - one(I)
        sstrt = D.sep.xadj[t]
        nr = rstop - rstrt + one(I)
        m = nr + D.sep.xadj[t + one(I)] - sstrt
        lxwgt[one(I)] = one(I)

        for x in oneto(m)
            if x <= nr
                g = rstrt + x - one(I)
            else
                g = D.sep.adjncy[sstrt + x - nr - one(I)]
            end

            lxwgt[x + one(I)] = lxwgt[x] + xwgt[g + one(I)] - xwgt[g]
        end

        if length(lwgt) < lxwgt[m + one(I)] - one(I)
            lwgt = Vector{eltype(wgt)}(undef, max(2 * length(lwgt), lxwgt[m + one(I)] - one(I)))
        end

        for x in oneto(m)
            if x <= nr
                g = rstrt + x - one(I)
            else
                g = D.sep.adjncy[sstrt + x - nr - one(I)]
            end

            l = lxwgt[x] - xwgt[g]

            for i in xwgt[g]:xwgt[g + one(I)] - one(I)
                lwgt[l + i] = wgt[i]
            end
        end
    end

    return lwgt
end
