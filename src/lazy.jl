#
# Lazy dynamic programming for the Boolean semiring.
#
# The eager search (eager.jl) computes, at each node, the table of every
# separator assignment that extends into the subtree, children first.
# The lazy search runs the other way: the search at a node asks each
# child whether the child's separator assignment extends into its
# subtree, the child answers by searching its own bag with the
# separator fixed, and the answers are memoized. Only the assignments
# the parent reaches are ever computed, and the search stops at the
# first homomorphism.
#
# The plan of every node is laid out in flat arrays. The element k of
# the bag of node t is the flat element
#
#    xbag[t] + k - 1,
#
# the residual of t is k = 1, ..., nres[t], and its separator is
# k = nres[t] + 1, ..., m.
#
struct LazyPlan{I}
    #
    # the bags
    #
    xbag::FVector{I}
    nres::FVector{I}
    vtx::FVector{I}
    #
    # the plan: where the candidates of each element come from, with
    # bpos a flat element
    #
    bknd::FVector{I}
    bpos::FVector{I}
    barc::FVector{I}
    #
    # the plan: the edge checks at each element, with chkp a flat
    # element
    #
    xchk::FVector{I}
    chka::FVector{I}
    chkp::FVector{I}
    chkd::FVector{Bool}
    #
    # the children asked at each element, once it is assigned
    #
    xask::FVector{I}
    ask::FVector{I}
    #
    # the children asked at each node, before its search
    #
    xini::FVector{I}
    ini::FVector{I}
    #
    # the flat element of each separator incidence of a child in the
    # bag of its parent
    #
    port::FVector{I}
    #
    # the search state of each element and each node
    #
    kind::FVector{I}
    arg::FVector{I}
    cur::FVector{I}
    stop::FVector{I}
    h::FVector{I}
    kpos::FVector{I}
    qpos::FVector{I}
    #
    # the stack of nodes
    #
    stack::FVector{I}
end

#
# The memo: the answer of each node to each separator assignment it
# was asked about. The entries live in an open-addressing hash table,
# and the assignment of entry e is
#
#    key[koff[e] + 1], ..., key[koff[e] + s]
#
# where s is the size of the separator of node[e].
#
mutable struct LazyMemo{I}
    slot::Vector{I}
    node::Vector{I}
    koff::Vector{I}
    ans::Vector{Bool}
    key::Vector{I}
    n::I
    nkey::I
end

function LazyMemo{I}() where {I}
    slot = zeros(I, 1024)
    node = Vector{I}(undef, 512)
    koff = Vector{I}(undef, 512)
    ans  = Vector{Bool}(undef, 512)
    key  = Vector{I}(undef, 1024)
    return LazyMemo{I}(slot, node, koff, ans, key, zero(I), zero(I))
end

#
# Search lazily and, unless witness is false, find a homomorphism by
# replaying each node with the separator assignment chosen by its
# parent.
#
function lazy(
        R::Boolean,
        D::Decomposition{I},
        Y::Instance{I},
        yord::AbstractVector{I},
        pre::Span{I},
        base::AbstractVector{I},
        typ::Span{I},
        xwgt,
        wgt,
        witness::Bool,
    ) where {I}
    if isnothing(wgt)
        T = valtype(R)
    else
        T = eltype(wgt)
    end

    nu = D.inst.dom.nu
    nt = D.tree.nu
    L = LazyPlan(D)
    M = LazyMemo{I}()
    stack = L.stack
    h = L.h
    #
    # the search: the node on top of the stack runs until it asks a
    # child that is not in the memo, which is pushed, or until it
    # answers, when it is popped and its answer is memoized and passed
    # to its parent
    #
    # ans is the answer to the pending ask of the node on top: 0 (none),
    # 1 (false), or 2 (true)
    #
    n = one(I)
    stack[n] = nt
    start!(L, nt)
    ans = zero(I)
    found = false

    while ispositive(n)
        t = stack[n]
        c = step!(L, M, D, Y, pre, base, typ, xwgt, wgt, t, ans)

        if ispositive(c)
            n += one(I)
            stack[n] = c
            start!(L, c)
            ans = zero(I)
        else
            found = c < zero(I)
            n -= one(I)

            if ispositive(n)
                memoize!(L, M, t, found)

                if found
                    ans = two(I)
                else
                    ans = one(I)
                end
            end
        end
    end

    if !found
        return szero(R, T), nothing
    end

    if !witness
        return sone(R, T), nothing
    end
    #
    # the witness: replay each node, top-down, with the separator
    # assignment chosen by its parent; every ask is in the memo
    #
    hg = FVector{I}(undef, nu)

    for t in nt:-one(I):one(I)
        b = L.xbag[t] - one(I)
        nr = L.nres[t]
        rstrt = D.res.xadj[t]
        sstrt = D.sep.xadj[t]
        sstop = D.sep.xadj[t + one(I)] - one(I)

        for p in sstrt:sstop
            h[b + nr + p - sstrt + one(I)] = hg[D.sep.adjncy[p]]
        end

        start!(L, t)
        c = step!(L, M, D, Y, pre, base, typ, xwgt, wgt, t, zero(I))
        @assert c < zero(I)

        for k in oneto(nr)
            hg[rstrt + k - one(I)] = h[b + k]
        end
    end

    out = FVector{I}(undef, nu)

    for r in oneto(nu)
        out[D.res.adjncy[r]] = yord[hg[r]]
    end

    return sone(R, T), out
end

#
# Lay out the plan of every node.
#
function LazyPlan(D::Decomposition{I}) where {I}
    nu = D.inst.dom.nu
    nt = D.tree.nu
    mbag, mchd, msep, medg = bagsizes(D)
    #
    # the contraction and plan of one node
    #
    pos     = FVector{I}(undef, nu)
    xadj    = FVector{I}(undef, mbag + one(I))
    adjncy  = FVector{I}(undef, medg)
    vtx     = FVector{I}(undef, mbag)
    arc     = FVector{I}(undef, medg)
    sxadj   = FVector{I}(undef, mchd + one(I))
    sadjncy = FVector{I}(undef, msep)
    lxchk   = FVector{I}(undef, mbag + one(I))
    lchka   = FVector{I}(undef, medg)
    lchkp   = FVector{I}(undef, medg)
    lchkd   = FVector{Bool}(undef, medg)
    lbknd   = FVector{I}(undef, mbag)
    lbpos   = FVector{I}(undef, mbag)
    lbarc   = FVector{I}(undef, mbag)
    #
    # the sizes
    #
    xbag = FVector{I}(undef, nt + one(I))
    nres = FVector{I}(undef, nt)
    xbag[one(I)] = one(I)
    nchk = zero(I)

    for t in oneto(nt)
        C = restrict!(D, pos, xadj, adjncy, vtx, arc, sxadj, sadjncy, t)
        xbag[t + one(I)] = xbag[t] + C.inst.dom.nu
        nres[t] = C.nr
        nchk += C.inst.dom.ne
    end

    ne = xbag[nt + one(I)] - one(I)
    ns = D.sep.xadj[nt + one(I)] - one(I)

    bvtx = FVector{I}(undef, ne)
    bknd = FVector{I}(undef, ne)
    bpos = FVector{I}(undef, ne)
    barc = FVector{I}(undef, ne)
    xchk = FVector{I}(undef, ne + one(I))
    chka = FVector{I}(undef, nchk)
    chkp = FVector{I}(undef, nchk)
    chkd = FVector{Bool}(undef, nchk)
    xask = FVector{I}(undef, ne + one(I))
    ask  = FVector{I}(undef, nt)
    xini = FVector{I}(undef, nt + one(I))
    ini  = FVector{I}(undef, nt)
    port = FVector{I}(undef, ns)
    #
    # the element of each child at which it is asked, or 0 if it is
    # asked before the search of its parent, and its parent
    #
    when = FVector{I}(undef, nt)
    prnt = FVector{I}(undef, nt)

    for e in oneto(ne + one(I))
        xask[e] = zero(I)
    end

    for t in oneto(nt + one(I))
        xini[t] = zero(I)
    end

    xchk[one(I)] = one(I)

    for t in oneto(nt)
        C = restrict!(D, pos, xadj, adjncy, vtx, arc, sxadj, sadjncy, t)
        plan!(C, lxchk, lchka, lchkp, lchkd, lbknd, lbpos, lbarc)
        b = xbag[t] - one(I)
        o = xchk[b + one(I)] - one(I)
        m = C.inst.dom.nu
        nr = C.nr

        for k in oneto(m)
            bvtx[b + k] = C.inst.vtx[k]
            bknd[b + k] = lbknd[k]
            bpos[b + k] = b + lbpos[k]
            barc[b + k] = lbarc[k]
            xchk[b + k + one(I)] = o + lxchk[k + one(I)]
        end

        for q in oneto(lxchk[m + one(I)] - one(I))
            chka[o + q] = lchka[q]
            chkp[o + q] = b + lchkp[q]
            chkd[o + q] = lchkd[q]
        end
        #
        # each child is asked at the smallest of its ports, the last
        # to be assigned, or before the search if its ports are all in
        # the separator
        #
        cstrt = D.tree.xadj[t]

        for j in oneto(C.sep.nu)
            c = D.tree.adjncy[cstrt + j - one(I)]
            pstrt = C.sep.xadj[j]
            pstop = C.sep.xadj[j + one(I)] - one(I)
            k = nr + one(I)

            for p in pstrt:pstop
                k = min(k, C.sep.adjncy[p])
                port[D.sep.xadj[c] + p - pstrt] = b + C.sep.adjncy[p]
            end

            prnt[c] = t

            if k <= nr
                when[c] = b + k

                if b + k < ne
                    xask[b + k + two(I)] += one(I)
                end
            else
                when[c] = zero(I)

                if t < nt
                    xini[t + two(I)] += one(I)
                end
            end
        end
    end

    xask[one(I)] = q = one(I)

    for e in oneto(ne)
        xask[e + one(I)] = q += xask[e + one(I)]
    end

    xini[one(I)] = q = one(I)

    for t in oneto(nt)
        xini[t + one(I)] = q += xini[t + one(I)]
    end

    for c in oneto(nt - one(I))
        e = when[c]

        if ispositive(e)
            ask[xask[e + one(I)]] = c
            xask[e + one(I)] += one(I)
        else
            t = prnt[c]
            ini[xini[t + one(I)]] = c
            xini[t + one(I)] += one(I)
        end
    end

    kind  = FVector{I}(undef, ne)
    arg   = FVector{I}(undef, ne)
    cur   = FVector{I}(undef, ne)
    stop  = FVector{I}(undef, ne)
    h     = FVector{I}(undef, ne)
    kpos  = FVector{I}(undef, nt)
    qpos  = FVector{I}(undef, nt)
    stack = FVector{I}(undef, nt)

    return LazyPlan{I}(xbag, nres, bvtx, bknd, bpos, barc, xchk, chka, chkp, chkd, xask, ask, xini, ini, port, kind, arg, cur, stop, h, kpos, qpos, stack)
end

#
# Start the search of node t, whose separator assignment is in h.
#
function start!(L::LazyPlan{I}, t::I) where {I}
    L.kpos[t] = L.nres[t] + one(I)
    L.qpos[t] = L.xini[t]
    return
end

#
# Run the search of node t, given the answer ans to its pending ask:
# 0 (none), 1 (false), or 2 (true). Returns a child c > 0 to ask, 0 if
# the separator assignment of t does not extend, or -1 if it does, with
# the residual assignment in h.
#
# The search state is k, the element whose asks are being made (nr + 1
# for the asks before the search), and q, the next ask.
#
function step!(
        L::LazyPlan{I},
        M::LazyMemo{I},
        D::Decomposition{I},
        Y::Instance{I},
        pre::Span{I},
        base::AbstractVector{I},
        typ::Span{I},
        xwgt,
        wgt,
        t::I,
        ans::I,
    ) where {I}
    h = L.h
    b = L.xbag[t] - one(I)
    nr = L.nres[t]
    k = L.kpos[t]
    q = L.qpos[t]
    rejected = isone(ans)

    if ans == two(I)
        q += one(I)
    end

    while true
        if !rejected
            #
            # the asks at k
            #
            if k > nr
                qstop = L.xini[t + one(I)] - one(I)
                lst = L.ini
            else
                qstop = L.xask[b + k + one(I)] - one(I)
                lst = L.ask
            end

            while q <= qstop
                c = lst[q]
                #
                # the separator assignment of c, read off h
                #
                lo = L.xbag[c] + L.nres[c] - one(I)
                pstrt = D.sep.xadj[c]
                pstop = D.sep.xadj[c + one(I)] - one(I)

                for p in pstrt:pstop
                    h[lo + p - pstrt + one(I)] = h[L.port[p]]
                end

                r = lookup(L, M, c)

                if iszero(r)
                    L.kpos[t] = k
                    L.qpos[t] = q
                    return c
                elseif isone(r)
                    rejected = true
                    break
                end

                q += one(I)
            end
        end

        if !rejected
            #
            # the asks passed: assign the next element
            #
            if isone(k) || iszero(nr)
                return -one(I)
            elseif k > nr
                k = nr
            else
                k -= one(I)
            end

            candidates!(L, Y, pre, base, typ, b + k)
        elseif k > nr
            return zero(I)
        end
        #
        # the next candidate at k, backtracking when they are exhausted
        #
        rejected = false

        while !advance!(L, D, Y, pre, typ, xwgt, wgt, t, b, k)
            k += one(I)

            if k > nr
                return zero(I)
            end
        end

        q = L.xask[b + k]
    end
end

#
# Start the candidates for the flat element e, in order of preference:
#
#   3. the image of an assigned element pointing to it
#   2. the preimages of an assigned element it points to
#   1. all elements of its type
#
@inline function candidates!(
        L::LazyPlan{I},
        Y::Instance{I},
        pre::Span{I},
        base::AbstractVector{I},
        typ::Span{I},
        e::I,
    ) where {I}
    kind = L.kind
    arg = L.arg
    cur = L.cur
    stop = L.stop
    kind[e] = L.bknd[e]

    if kind[e] == two(I) + one(I)
        arg[e] = image(Y, L.h[L.bpos[e]], L.barc[e])
        cur[e] = one(I)
        stop[e] = one(I)
    elseif kind[e] == two(I)
        a = L.barc[e]
        l = base[a] + L.h[L.bpos[e]] - typ.xadj[Y.cod.adjncy[a]]
        cur[e] = pre.xadj[l]
        stop[e] = pre.xadj[l + one(I)] - one(I)
    else
        u = L.vtx[e]
        cur[e] = typ.xadj[u]
        stop[e] = typ.xadj[u + one(I)] - one(I)
    end

    return
end

#
# Assign the next valid candidate to the element k of node t: one with
# nonzero weight, consistent with every edge to an element assigned
# earlier. Returns false when the candidates are exhausted.
#
@inline function advance!(
        L::LazyPlan{I},
        D::Decomposition{I},
        Y::Instance{I},
        pre::Span{I},
        typ::Span{I},
        xwgt,
        wgt,
        t::I,
        b::I,
        k::I,
    ) where {I}
    e = b + k
    kind = L.kind
    cur = L.cur
    stop = L.stop
    h = L.h

    while cur[e] <= stop[e]
        c = cur[e]
        cur[e] += one(I)

        if isone(kind[e])
            y = c
        elseif kind[e] == two(I)
            y = pre.adjncy[c]
        else
            y = L.arg[e]
        end

        if !isnothing(wgt)
            g = D.res.xadj[t] + k - one(I)
            l = xwgt[g]

            if l < xwgt[g + one(I)] && iszero(wgt[l + y - typ.xadj[L.vtx[e]]])
                continue
            end
        end

        h[e] = y

        if consistent(Y, h, L.xchk, L.chka, L.chkp, L.chkd, y, e)
            return true
        end
    end

    return false
end

#
# The hash of the separator assignment of node c: the values
# v[o + 1], ..., v[o + s].
#
@inline function fingerprint(c::I, v::AbstractVector{I}, o::I, s::I) where {I}
    x = UInt64(c) * 0x9e3779b97f4a7c15

    for i in oneto(s)
        x = (x ⊻ UInt64(v[o + i])) * 0xbf58476d1ce4e5b9
    end

    return x ⊻ (x >>> 31)
end

#
# The memoized answer of node c to its separator assignment in h: 0
# (none), 1 (false), or 2 (true).
#
function lookup(L::LazyPlan{I}, M::LazyMemo{I}, c::I) where {I}
    o = L.xbag[c] + L.nres[c] - one(I)
    s = L.xbag[c + one(I)] - one(I) - o
    h = L.h
    slot = M.slot
    mask = length(slot) - one(I)
    i = (fingerprint(c, h, o, s) % I) & mask

    while true
        e = slot[i + one(I)]

        if iszero(e)
            return zero(I)
        end

        if M.node[e] == c
            l = M.koff[e]
            same = true

            for j in oneto(s)
                if M.key[l + j] != h[o + j]
                    same = false
                    break
                end
            end

            if same
                if M.ans[e]
                    return two(I)
                else
                    return one(I)
                end
            end
        end

        i = (i + one(I)) & mask
    end
end

#
# Memoize the answer r of node c to its separator assignment in h,
# which is not in the memo. The table doubles at half load.
#
function memoize!(L::LazyPlan{I}, M::LazyMemo{I}, c::I, r::Bool) where {I}
    o = L.xbag[c] + L.nres[c] - one(I)
    s = L.xbag[c + one(I)] - one(I) - o

    if 2 * (M.n + one(I)) > length(M.slot)
        grow!(L, M)
    end

    n = M.n += one(I)

    if length(M.node) < n
        resize!(M.node, 2 * length(M.node))
        resize!(M.koff, 2 * length(M.koff))
        resize!(M.ans, 2 * length(M.ans))
    end

    if length(M.key) < M.nkey + s
        resize!(M.key, max(2 * length(M.key), M.nkey + s))
    end

    M.node[n] = c
    M.koff[n] = M.nkey
    M.ans[n] = r

    for j in oneto(s)
        M.key[M.nkey + j] = L.h[o + j]
    end

    M.nkey += s
    place!(M, n, fingerprint(c, L.h, o, s))
    return
end

function place!(M::LazyMemo{I}, n::I, x::UInt64) where {I}
    slot = M.slot
    mask = length(slot) - one(I)
    i = (x % I) & mask

    while !iszero(slot[i + one(I)])
        i = (i + one(I)) & mask
    end

    slot[i + one(I)] = n
    return
end

function grow!(L::LazyPlan{I}, M::LazyMemo{I}) where {I}
    M.slot = zeros(I, 2 * length(M.slot))

    for n in oneto(M.n)
        c = M.node[n]
        s = L.xbag[c + one(I)] - L.xbag[c] - L.nres[c]
        place!(M, n, fingerprint(c, M.key, M.koff[n], s))
    end

    return
end
