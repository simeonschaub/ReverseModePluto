### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ d5559dba-9fe4-11ec-3744-ebd1408e7dc4
using AbstractTrees, PlutoUI, HypertextLiteral, PlutoTest, LinearAlgebra, Plots

# ╔═╡ 4da7a594-aec5-4c8d-ab20-6913ed6ff8ac
TableOfContents(title="📚 Table of Contents", indent=true, depth=4, aside=true)

# ╔═╡ 1b768306-0e0e-414d-a501-5c62a2db720f
md"## Computations as Trees"

# ╔═╡ 6733267e-1251-49a7-86a0-ff45ba408205
md"### Tracking variables"

# ╔═╡ 17f9e2c0-aa70-4d4e-ba42-51782df43785
md"### Computational Tree (Forward Pass)"

# ╔═╡ 7299bfea-e23e-4993-bebd-9d8762427294
md"### Pretty Viz: Forward Pass + Reverse Mode Diff"

# ╔═╡ be693f92-de60-4b43-82b3-ee6cceea310a
md"""

$\frac{\partial}{\partial x} (2xy + (x-1)^2) = 2y + 2(x-1)$
$\frac{\partial}{\partial y} (2xy + (x-1)^2) = 2x$

If x = 3 and y = 5, then these evaluate to  14 and 6.
"""

# ╔═╡ 20571e1a-9687-4d42-98b3-b3bc0b207b3b
md"## Implementation in Code"

# ╔═╡ 5e02b2c0-10cf-4746-840d-7017648f89f8
md"""
Besides the mathematics of reverse mode differentiation, there are many interesting computer science issues regarding the implementation of reverse diff.  Probably the jury is still out which is best.

The implementation below is based on Julia's dispatch system.
This means we create a type `Tracked` for keeping track of our input variables and everything we'll need to calculate later.

The other main approach (not illustrated here) is source to source translation of code. In Julia the zygote package translates at the high level, while the increasingly popular Enzyme package translates at the lower level.
"""

# ╔═╡ d1312080-272c-4bd5-bcfe-bf47f8320961
md"### The Tracked Data Structure"

# ╔═╡ 99b6ab91-a022-449c-988c-0e5c5719c910
begin

	struct Tracked{T} <: Number
		val::T
		name::Symbol
		df
		deps::Vector{Tracked}
	end

	# the two argument version as in Tracked{Int}(3,:x)
	Tracked{T}(x, name=gensym()) where {T} = Tracked{T}(x, name, nothing, Tracked[])

	## The above is typical Julia, the below shows up often when you 
	## define a new number type and you want ordinary numbers to work together
	## with your newly defined numbers.
	## This is similar to when in mathematics, strictly speaking integers
	## are separate from the reals, but we all treat integers as embedded
	## in the reals.

	# Converting a Tracked variable, converts like the number
	Base.convert(T::Type{Tracked{S}}, x::Tracked) where {S} = T(convert(S, x.val), x.name, x.df, x.deps)

	# This tells Julia to convert any number added to a `Tracked` to a `Tracked` first
	Base.promote_rule(::Type{Tracked{S}}, ::Type{T}) where {S<:Number, T<:Number} = Tracked{promote_type(S, T)}

	# All this is just for nicer printing
	function Base.show(io::IO, x::Tracked)
		if x.df === nothing
			print(io, Base.isgensym(x.name) ? x.val : "$(x.val)=$(x.name)")
		else
			#print(io, "⋅Tracked(")
			print(io,"[")
			print(io, x.name)
			print(io,"]")
			#print(io, ", ")
			 print(io," → ")
			show(io, x.val)
			#print(io, ")")
		end
	end
	function Base.show(io::IO, x::Tracked{<:AbstractArray})
		io = IOContext(io, :compact=>true)
		print(io, Base.isgensym(x.name) ? x.val : string(x.name))
	end
	Base.show(io::IO, ::MIME"text/plain", x::Tracked) = print_tree(io, x)

	Base.:(==)(x::Tracked, y::Tracked) = x === y
end

# ╔═╡ 0b5e6560-81fd-4182-bba5-aca702fb3048
begin
   # The second argument of Tracked is a symbol only used for printing
   x = Tracked{Int}(3, :x)
   y = Tracked{Int}(5, :y)
   u = Tracked{Int}(2, :u)
end

# ╔═╡ 0d006e4d-72e3-42ab-beb3-e5304a4d9ba9
x

# ╔═╡ 8919faec-753d-41d0-b0a8-c84c025b2919
xfloat = convert( Tracked{Float64}, x)

# ╔═╡ be28f764-a415-46ed-a503-68e8d831f8a8
(aa,bb) = (promote(x,2.5)) # promotes x to be a float and also y to be a Tracked

# ╔═╡ a0c0f48d-e0b0-4f45-8e0e-d9ab5743958b
Dump(aa)

# ╔═╡ 814786be-a760-4023-a614-83e727681075
Dump(bb)

# ╔═╡ 2ba83769-89b4-4ba0-8726-9210f394b4e7
md"### Operations on Tracked"

# ╔═╡ 885cd51d-895f-4996-a23b-780498b5b810
md"""
All overloads will do the operation (e.g. sum `x` and `y`), but also remember the pullback map and input variables for the reverse pass.
"""

# ╔═╡ 4813e57f-557a-4179-afd2-7925687b3d35
## Simeon write your way too
# These operations are

# d(x/y)= (y dx - x dy)/y^2
# ∇(x/y) = [1/y; -x/y^2] # Notice that the Δ -> (?,?) is just the gradient

# ╔═╡ ea94bdf9-b0b1-42ad-ba44-aaba73ed1192
md"""
Let's look at our example from above, $z = x^2+y$.
We'll introduce an intermediate variable $u = x^2$ so we can write $z$ as $u + y$.

We will first start with $1 \cdot dz$. The one here represents the fact that $\frac{\partial z}{\partial z} = 1$.

We then want to *pull* that *back* until we get something in terms of $dx$ and $dy$, so we then get the derivatives wrt $x$ and $y$

The pullback map for $+$ will first map $dz$ to $d(u + y) = du + dy$ and then we can use the pullback map for ^ to expand $du$ to $2x \cdot dx$.

Using that recursive method we end up with $dz = 2x \cdot dx + dy$, which is exactly what we'd get if we did this by hand. This tells us that $\frac{\partial z}{\partial x} = 2x$ and $\frac{\partial z}{\partial y} = y$.
"""

# ╔═╡ 13487e65-5e48-4a37-9bea-f262dd7b6d56
function Base.:+(x::Tracked, y::Tracked)
	Tracked(x.val + y.val, :+, Δ -> (Δ, Δ), Tracked[x, y])
end

# ╔═╡ c9d70049-4793-41db-9075-e9e73e926c1a
x+y

# ╔═╡ b0cc4665-eb45-48ea-9a33-5acf56d2a283
function Base.:-(x::Tracked, y::Tracked)
	Tracked(x.val - y.val, :-, Δ -> (Δ, -Δ), Tracked[x, y])
end

# ╔═╡ 73d638bf-30c1-4694-b3a8-4b29c5e3fa65
function Base.:*(x::Tracked, y::Tracked)
	Tracked(x.val * y.val, :*, Δ -> (Δ * y.val', x.val' * Δ), Tracked[x, y])
end

# ╔═╡ 95ab6c83-0757-40b3-a116-6d872a074f69
x*y

# ╔═╡ db4f871a-3ca1-4bd3-a2c5-da7d6f599281
(x*y).val

# ╔═╡ 29360c2a-c47b-48a3-b225-248eb3ec8a42
(x*y).name

# ╔═╡ b47754ac-cb17-4ea3-a97d-20371fd341a5
((x*y).df)(1) # For reference

# ╔═╡ bb75e790-a628-4821-b7b1-11cd34a73d38
(x*y).deps

# ╔═╡ ac097299-0a31-474c-ab26-a4fb24bb9046
function Base.:^(x::Tracked, n::Int)
	Tracked(x.val^n, Symbol("^$n"), Δ -> (Δ * n * x.val^(n-1),), Tracked[x,])
end

# ╔═╡ aa99b123-7164-4ed7-833f-b99b19118a0d
x^2+y

# ╔═╡ 0d2a3187-46f1-4705-bb3e-ecf74e34840d
z = (2x*y + (x-1)^2)

# ╔═╡ 61513f88-c126-43ab-9515-85a2c8a472bf
(x^2+y).deps

# ╔═╡ 2c3f3103-9ccb-446e-a289-87bbd0a1b97b
(x^2+y).deps[2]

# ╔═╡ 795c510e-6280-42d9-8820-290ca07be88d
((x^2+y).deps[2]).deps

# ╔═╡ e450329b-46ee-4f53-b6f0-891d6596938c
x^2+y

# ╔═╡ d48b9f23-b1b8-43f8-a05d-2234b69bf1f0
(2x*y + (x-1)^2)

# ╔═╡ 2141849b-675e-406c-8df4-34b2706507af
function Base.:/(x::Tracked, y::Tracked)
	Tracked(x.val / y.val, :/, Δ -> (Δ / y.val, -Δ * x.val / y.val^2), Tracked[x, y])
end

# ╔═╡ 8ab0f55d-a393-4a8a-a48c-9ced26033f57
function Base.sin(x::Tracked)
	Tracked(sin(x.val), :sin, Δ -> (Δ * cos(x.val),), Tracked[x,])
end

# ╔═╡ 4bfc2f7d-a5b0-44c7-8bb6-f1b834c1cc51
md"""
`Tracked` is a tree -- We just need to tell *AbstractTrees.jl* how to get the children for each node and we get tree printing and iteration over all nodes for free.
"""

# ╔═╡ 2188a663-5a85-4ce4-bc8d-20383481e59b
AbstractTrees.children(x::Tracked) = x.deps

# ╔═╡ ccafd0b9-95aa-4e58-8afc-26cd3ee61cc9
md"""
Straight away we get the primal result of our calculation:
"""

# ╔═╡ 81eb8a2d-a3a9-45af-a5a5-b96aefd48712
(2x*y + (x-1)^2).val # The result of `2x*y + (x-1)^2`

# ╔═╡ 01eacbd4-ef37-4524-aecc-2ef9a1044cf8
md"""
To also get the gradient, we'll use `PreOrderDFS` to traverse the tree we just created from the top down.
"""

# ╔═╡ e52aa672-69a9-419b-a992-e7a3d1364fb6
# `PreOrderDFS` traverses this tree from the top down
Text.(collect(PreOrderDFS(z)))

# ╔═╡ d5565717-239b-41ab-b311-d59b383130ed
md"""
Ok, let's create our function `grad` which will accumulate all intermediate gradients into a dictionary:
"""

# ╔═╡ 99a3507b-ca03-429f-acde-e2d1ebb32054
function grad(f::Tracked)
	d = Dict{Any, Any}(f => 1)
	for x in PreOrderDFS(f) # recursively traverse all dependents
		x.df === nothing && continue # ignore untracked variables like constants
		dy = x.df(d[x]) # evaluate pullback
		for (yᵢ, dyᵢ) in zip(x.deps, dy)
			# store the gradient in d
			# if we have already stored a gradient for this variable, we need to add them
			d[yᵢ] = get(d, yᵢ, 0) .+ dyᵢ
		end
	end
	return d
end

# ╔═╡ d4e9b202-242e-4420-986b-12d2ab57af93
grad(f::Tracked, x::Tracked) = grad(f)[x]

# ╔═╡ 7fcccb65-4a7b-4527-97be-d25f481f6eaf
md"""
We can verify that it does the right thing:
"""

# ╔═╡ 18b1c55d-a6b5-44f6-b0b3-50bdb0aa9d96
w = x*y + x

# ╔═╡ 506d408e-dc2b-4e12-b917-286e3f4079a2
grad(w)

# ╔═╡ a7c8cb6a-6e17-4d8f-8958-fe3527c5b8e7
grad(w, x), grad(w, y)

# ╔═╡ e55dfad2-db50-459f-ab54-fa7637fc3638
md"""
## How can we visualize both the forward and the reverse pass?

We can further visualize each steps we just took. First we do the forwards calculation, where we also build up our tree, then we go down the tree in the opposite direction to accumulate our gradient.
"""

# ╔═╡ d9304d8d-9a34-46f2-908d-42d5cc5f5c5f
👽 = Tracked{Int}(12, :👽)

# ╔═╡ a6c2d3a2-326a-41dd-864d-aa3662466222
x,y

# ╔═╡ bcff2aa8-2387-44c7-a28f-39cd505a7adf
md"""
We can also visualize what Julia does in the forward pass on the code itself:
"""

# ╔═╡ 53ae4070-8818-4d21-8648-19df9319918a
md"""
## Let's write our own neural network!
"""

# ╔═╡ 8a70e835-dae6-4727-8204-95c87d5c23da
md"""
We need some more overloads
"""

# ╔═╡ 520f280d-c78e-433b-a0a2-8ef05b04f7cc
function Base.broadcasted(::typeof(tanh), x::Tracked)
	res = tanh.(x.val)
	Tracked(res, :tanh, Δ -> (Δ .* (1 .- res.^2),), Tracked[x,])
end

# ╔═╡ a5671f2f-48ca-4896-9428-147dc671d2b9
function Base.broadcasted(::typeof(+), x::Tracked{<:VecOrMat}, y::Tracked{<:Vector})
	Tracked(x.val .+ y.val, :.+, Δ -> (Δ, sum(Δ; dims=2)), Tracked[x, y])
end

# ╔═╡ e01114fa-a847-424a-931f-8f42e623109f
function LinearAlgebra.norm(x::Tracked)
	res = norm(x.val)
	Tracked(res, :norm, Δ -> (Δ/res .* x.val,), Tracked[x,])
end

# ╔═╡ e6ee1921-57aa-4194-b2eb-8e32fa4a6c44
macro t(s)
	:(Tracked{typeof($(esc(s)))}($(esc(s)), $(QuoteNode(s))))
end

# ╔═╡ 266bf2ed-d625-47be-8a87-21e12052b27f
md"""
Let's use three dense layers, all of the form

$x \mapsto \tanh.(W \cdot x + b)$
"""

# ╔═╡ 788903f0-a372-4265-bbf3-5b507a705100
NN(x, (W1, W2, W3, b1, b2, b3)) = W3 * tanh.(W2 * tanh.(W1 * @t(x) .+ b1) .+ b2) .+ b3

# ╔═╡ 337bc837-174f-4ed6-bee1-c9e80dc03b52
input = collect(range(0, 2π; length=100))'

# ╔═╡ 8f1ec8d7-89ab-4bc5-9eef-2b0a69019854
md"""
We'll try to approximate a sine curve:
"""

# ╔═╡ d2d21d41-9a4e-428b-b696-6878901849ba
ŷ = sin.(input) #.+ .2input .^ 2 .- cos.(5input)

# ╔═╡ 98807f0d-f8a5-4fe0-9983-2b95290347d9
md"""
### Tada! 🎉

(It's even reasonably fast for neural nets)
"""

# ╔═╡ 85098c31-04a1-421c-b3f6-25fec9360588
md"### What if we use a (leaky) rectified linear unit (ReLU) as activation?"

# ╔═╡ 7dca2c78-6faf-450a-8096-1c5b9f7f1295
leaky_relu(x) = x > 0 ? x : .1x

# ╔═╡ b2622dc4-2fd3-4d80-8efd-a6a985a6b2e9
function Base.broadcasted(::typeof(leaky_relu), x::Tracked)
	Tracked(leaky_relu.(x.val), :tanh, Δ -> (map((x, Δ) -> x > 0 ? Δ : .1Δ, x.val, Δ),), Tracked[x,])
end

# ╔═╡ 0911a08a-4290-455b-9b26-0bf2862296da
plot(leaky_relu; label="leaky relu")

# ╔═╡ 0b094198-cf44-41d7-a8dc-fd8fd0716bb4
md"""
---
### Helper Functions
"""

# ╔═╡ d82adc20-4c8c-4f2c-9839-d03ad7e7f581
begin
	struct EX
		x::Any
		function EX(ex)
			if Meta.isexpr(ex, :call) && ex.args[1] === :+ && length(ex.args) > 3
				new(Expr(:call, :+, Expr(:call, :+, ex.args[2:end-1]...), ex.args[end]))
			else
				new(ex)
			end
		end
	end
	show_tree(ex::Expr) = show_tree(EX(ex))
	function Base.show(io::IO, ex::EX)
		Base.show_unquoted(io, Meta.isexpr(ex.x, :call) ? ex.x.args[1] : ex.x)
		if Meta.isexpr(ex.x, :call) && ex.x.args[1] === :^
			print(io, ex.x.args[3])
		end
	end
	function AbstractTrees.children(ex::EX)
		if Meta.isexpr(ex.x, :call)
			ex.x.args[1] === :^ ? [EX(ex.x.args[2])] : EX.(ex.x.args[2:end])
		else
			EX[]
		end
	end
	Base.:(==)(ex1::EX, ex2::EX) = ex1.x == ex2.x
	Base.hash(ex::EX, i::UInt) = hash(ex.x, i)
end

# ╔═╡ 96286b65-1a22-4458-a399-46579248cce4
begin
	_rand(s...) = randn(s...) / √prod(s)
	W1, W2, W3 = _rand(32, 1), _rand(32, 32), _rand(1, 32)
	b1, b2, b3 = _rand(32), _rand(32), _rand(1)
	params = [@t(W1), @t(W2), @t(W3), @t(b1), @t(b2), @t(b3)]
	p = @animate for i in 1:10000
		loss = norm(NN(input, params) - @t(ŷ))
		∇ = grad(loss)
		for p in params
			p.val .-= 1e-3 .* ∇[p]
		end
		plot(input', [NN(input, params).val; ŷ]'; label=["prediction" "training data"])
	end every 200
	gif(p; fps=5)
end

# ╔═╡ 49333dc7-b809-4bb1-b285-20a12d8b9209
let ŷ_t = @t(ŷ)

	loss = norm(NN(input, params) - ŷ_t)
	∇ = grad(loss)
	params′ = deepcopy(params)
	for (p, p′) in zip(params, params′)
		p′.val .-= 1e-3 .* ∇[p]
	end

	plot(input', [NN(input, params).val; ŷ]'; label=["prediction" "training data"])
	quiver!(input[1:4:end], NN(input, params).val[1:4:end]; quiver=(zero(input[1:4:end]), 3(NN(input, params′).val .- NN(input, params).val)[1:4:end]))
end

# ╔═╡ 86fa378b-815d-4c3d-9121-1338ee54f30f
let
	NN(x, (W1, W2, W3, b1, b2, b3)) = W3 * leaky_relu.(W2 * leaky_relu.(W1 * @t(x) .+ b1) .+ b2) .+ b3

	_rand(s...) = randn(s...) / √prod(s)
	W1, W2, W3 = _rand(32, 1), _rand(32, 32), _rand(1, 32)
	b1, b2, b3 = _rand(32), _rand(32), _rand(1)
	params = [@t(W1), @t(W2), @t(W3), @t(b1), @t(b2), @t(b3)]
	p = @animate for i in 1:10000
		loss = norm(NN(input, params) - @t(ŷ))
		∇ = grad(loss)
		for p in params
			p.val .-= 1e-3 .* ∇[p]
		end
		plot(input', [NN(input, params).val; ŷ]'; label=["prediction" "training data"])
	end every 200
	gif(p; fps=5)
end

# ╔═╡ 8110f306-a7bb-43a2-bb36-6182c59b4b2e
begin
	struct TTREE
		x
		d::Dict{Any, Any}
	end
	function Base.show(io::IO, x::TTREE)
		show(io, x.x)
		print(io, " ")
		show(io, MIME("text/html"), get(x.d, x.x, @htl("")))
	end
	AbstractTrees.children(x::TTREE) = (TTREE(i, x.d) for i in children(x.x))
end

# ╔═╡ 1f1b384a-6588-45a5-9dd3-6de3face8bfb
function ad_steps(x::Expr; color_fwd="red", color_bwd="green", font_size=".8em")
	x = EX(x)
	repr(x) = sprint(show, x; context=:compact=>true)
	span_fwd = @htl "<span style='color: $color_fwd; font-size: $font_size'>"
	span_bwd = @htl "<span style='color: $color_bwd; font-size: $font_size'>"

    d1 = Dict(
		let e = eval(i.x)
			i => @htl "&ensp;$(span_fwd)$(repr(e isa Tracked ?  e.val : e))</span>"
		end
		for i in PostOrderDFS(x) if isempty(children(i))
	)

	res = accumulate(Iterators.filter(x -> !isempty(children(x)), PostOrderDFS(x)); init=d1) do d,i
		d = copy(d)
		e = eval(i.x)
		d[i] = @htl "&ensp;$(span_fwd)$(repr(e isa Tracked ?  e.val : e))</span>" 
		d
	end

	pushfirst!(res, d1)

	f = eval(x.x)
	d = Dict{Any, Any}(f => 1)
	let d1 = copy(res[end])
		d1[x] = @htl "$(d1[x])&ensp;$(span_bwd)1</span>"
		push!(res, d1)
	end
	for (x, e) in zip(PreOrderDFS.((f, x))...)
		x.df === nothing && continue
		dy = x.df(d[x])
		for (yᵢ, dyᵢ, e) in zip(x.deps, dy, children(e))
			d1 = copy(res[end])
			if haskey(d, yᵢ)
				d1[e] = @htl "$(get(d1, e, ""))$(span_bwd) + $(repr(dyᵢ))</span>"
			else
				d1[e] = @htl "$(get(d1, e, ""))&ensp;$(span_bwd)$(repr(dyᵢ))</span>"
			end
			push!(res, d1)
			d[yᵢ] = get(d, yᵢ, 0) + dyᵢ
		end
	end
	res
end

# ╔═╡ 1a154bb7-93a3-4973-8908-788db77ac294
s2 = @htl """
<link rel="stylesheet" href="https://fperucic.github.io/treant-js/Treant.css"/>
<style>
.Treant > .node {
	padding: 5px; border: 2px solid #484848; border-radius: 8px;
	box-sizing: unset;
	min-width: fit-content;
	font-size: 1.6em;
}
.Treant > .node > span {
	vertical-align: middle;
}

.Treant .collapse-switch { width: 100%; height: 100%; border: none; }
.Treant .node.collapsed { background-color: var(--main-bg-color); }
.Treant .node.collapsed .collapse-switch { background: none;}
</style>

<script src="https://fperucic.github.io/treant-js/vendor/jquery.min.js"></script>
<script src="https://fperucic.github.io/treant-js/vendor/jquery.easing.js"></script>
<script src="https://fperucic.github.io/treant-js/vendor/raphael.js"></script>
<script src="https://fperucic.github.io/treant-js/Treant.js"></script>
"""

# ╔═╡ 6b1fb808-e993-4c2b-b81b-6710f8206de7
function to_json(x)
	d = Dict{Symbol, Any}(
		:innerHTML => sprint(AbstractTrees.printnode, x),
		:children => Any[to_json(c) for c in children(x)],
		#:collapsed => !isempty(children(x)),
	)
end

# ╔═╡ 437285d4-ec53-4bb7-9966-fcfb5352e205
function show_tree(x; height=400)
	id = gensym()
	@htl """
	$s2
	<div id="$id" style="width:100%; height: $(height)px"> </div>
	<script>
	var simple_chart_config = {
		chart: {
			container: "#$id",

			//animateOnInit: true,

			node: {
				collapsable: true,
			},

			nodeAlign: "BOTTOM",

			connectors: {
				type: "straight",
				style: {
					stroke: getComputedStyle(document.documentElement).getPropertyValue('--cm-editor-text-color')
				}
			},
			animation: {
				nodeAnimation: "easeOutBounce",
				nodeSpeed: 500,
				connectorsAnimation: "bounce",
				connectorsSpeed: 500
			},
		},

		nodeStructure: $(to_json(x))
	};
	var my_chart = new Treant(simple_chart_config);
	</script>
	"""
end

# ╔═╡ 86aa821b-a373-4814-953e-535f3a33c002
show_tree(norm(NN(input, params) - @t(ŷ)); height = 1000)

# ╔═╡ 094bf7ee-7c9b-458c-98dc-521768831654
function show_steps(steps; height=400)
	id1, id2 = gensym(), gensym()
	@htl """
	$s2
	<div id="$id1" style="width:100%; height: $(height)px"> </div>
	<span style="color: red; font-size: 1.5em; margin-bottom: 1em">Move me!</span>
	<input id="$id2" type="range" min="0" max="$(length(steps)-1)" value="0" style="width: 100%">
	<script>
	var steps = $(to_json.(steps));
	var simple_chart_config = {
		chart: {
			container: "#$id1",

			//animateOnInit: true,

			node: {
				collapsable: true,
			},

			nodeAlign: "BOTTOM",

			connectors: {
				type: "straight",
				style: {
					stroke: getComputedStyle(document.documentElement).getPropertyValue('--cm-editor-text-color')
				}
			},
			animation: {
				nodeAnimation: "easeOutBounce",
				nodeSpeed: 500,
				connectorsAnimation: "bounce",
				connectorsSpeed: 500
			},
		},
		nodeStructure: steps[0]
	};
	var my_chart = new Treant(simple_chart_config, null, \$);
	var slider = document.getElementById("$id2");
	slider.addEventListener('input', function(){
		simple_chart_config.nodeStructure = steps[this.value];
		document.getElementById("$id1").innerHTML = "";
		my_chart = new Treant(simple_chart_config, null, \$);
	});
	</script>
	"""
end

# ╔═╡ faf69c6f-b179-4ca2-aa22-641e9560e17b
macro ad_steps(ex)
	quote
		ex = $(QuoteNode(ex))
		steps = ad_steps(ex)
		show_steps([TTREE(EX(ex), step) for step in steps])
	end
end

# ╔═╡ 27b39d7d-fc08-4ccc-aea4-b64f8a4f5726
#@ad_steps 3y*x + 2(x-1)*x
@ad_steps (2x*y + (x-1)^2)  # type the steps for the macro

# ╔═╡ ec723b90-1c8b-4a87-8a86-fe17d6e85f22
@ad_steps (x+y)

# ╔═╡ 0d186a1b-eb89-4be8-a237-e5a9608f4a30
@ad_steps (x*y)

# ╔═╡ db3389c0-cb00-404c-81a8-8c4ccdab22f4
@ad_steps ((x*y)*u)

# ╔═╡ 8b486bd8-25a1-4895-99c8-0541c9e3c8b0
@ad_steps (x/y)

# ╔═╡ 2eb6af9c-9ac7-4b64-93fb-c5a2f037d303
@ad_steps y^2

# ╔═╡ f6ce8448-d9ce-4453-9e47-dc6443d50f55
s1 = html"""
<style>
p-frame-viewer {
	display: inline-flex;
	flex-direction: column;
}
p-frames,
p-frame-controls {
	display: inline-flex;
}
p-frame-controls {
	margin-top: 20px;
}
line-like {
	font-size: 30px;
}
"""

# ╔═╡ 9a141034-17cb-4d85-a5a2-4724a38dd269
macro visual_debug(expr)
	s1
	quote
		$(esc(:(PlutoTest.@eval_step_by_step($expr)))) .|> PlutoTest.SlottedDisplay |> PlutoTest.frames |> PlutoTest.with_slotted_css
	end
end

# ╔═╡ 79f71f9d-b491-4a2c-85a4-29ae8da4f312
@visual_debug 3y + 2(x-1)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AbstractTrees = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTest = "cb4044da-4d16-4ffa-a6a3-8cad7f73ebdc"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
AbstractTrees = "~0.4.2"
HypertextLiteral = "~0.9.4"
Plots = "~1.31.7"
PlutoTest = "~0.2.2"
PlutoUI = "~0.7.39"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.0"
manifest_format = "2.0"
project_hash = "eceb496e896aa6e7e79fc43936b5726532da0604"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.AbstractTrees]]
git-tree-sha1 = "5c0b629df8a5566a06f5fef5100b53ea56e465a0"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.2"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "80ca332f6dcb2508adba68f22f551adb2d00a624"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.3"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "1fd869cc3875b57347f7027521f561cf46d1fcd8"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.19.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "5856d3031cdb1f3b2b6340dfdc66b6d9a149a374"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.2.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "5158c2b41018c5f7eb1470d558127ac274eca0c9"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.1"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.Extents]]
git-tree-sha1 = "5e1e4c53fa39afe63a7d356e30452249365fba99"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.1"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "ccd479984c7838684b3ac204b716c89955c76623"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "cf0a9940f250dc3cb6cc6c6821b4bf8a4286cf9c"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.66.2"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "2d908286d120c584abbe7621756c341707096ba4"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.66.2+0"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "fb28b5dc239d0174d7297310ef7b84a11804dfab"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.0.1"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "a7a97895780dab1085a97769316aa348830dc991"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.3"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "Dates", "IniFile", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "bf7fcabde6565fd8226d8a4c31bc8bb25fdd7f1d"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.3.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "1a43be956d433b5d0321197150c2f94e16c0aaa0"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.16"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "94d9c52ca447e23eac0c0f074effbcd38830deb5"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.18"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "5d4d2d9904227b8bd66386c1138cf4d5ffa826bf"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.9"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "d9ab10da9de748859a7780338e1d6566993d1f25"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e60321e3f2616584ff98f0a4f18d98ae6f89bbb3"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.17+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0044b23da09b5608b4ecacb4e5e6c6332f833a7e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.2"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "8162b2f8547bc23876edd0c5181b27702ae58dce"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.0.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "9888e59493658e476d3073f1ce24348bdc086660"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "a19652399f43938413340b2068e11e55caa46b65"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.31.7"

[[deps.PlutoTest]]
deps = ["HypertextLiteral", "InteractiveUtils", "Markdown", "Test"]
git-tree-sha1 = "17aa9b81106e661cffa1c4c36c17ee1c50a86eda"
uuid = "cb4044da-4d16-4ffa-a6a3-8cad7f73ebdc"
version = "0.2.2"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8d1f54886b9037091edf146b517989fc4a09efec"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.39"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "e7eac76a958f8664f2718508435d058168c7953d"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.3"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "22c5201127d7b243b9ee1de3b43c408879dff60f"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.3.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "dfec37b90740e3b9aa5dc2613892a3fc155c3b42"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.6"

[[deps.StaticArraysCore]]
git-tree-sha1 = "ec2bd695e905a3c755b33026954b119ea17f2d22"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.3.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArraysCore", "Tables"]
git-tree-sha1 = "8c6ac65ec9ab781af05b08ff305ddc727c25f680"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.12"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "ed5d390c7addb70e90fd1eb783dcb9897922cbfa"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.8"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "e59ecc5a41b000fa94423a578d29290c7266fc10"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "58443b63fb7e465a8a7210828c91c08b92132dff"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.14+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╠═d5559dba-9fe4-11ec-3744-ebd1408e7dc4
# ╠═4da7a594-aec5-4c8d-ab20-6913ed6ff8ac
# ╟─1b768306-0e0e-414d-a501-5c62a2db720f
# ╟─6733267e-1251-49a7-86a0-ff45ba408205
# ╠═0b5e6560-81fd-4182-bba5-aca702fb3048
# ╟─17f9e2c0-aa70-4d4e-ba42-51782df43785
# ╠═c9d70049-4793-41db-9075-e9e73e926c1a
# ╠═aa99b123-7164-4ed7-833f-b99b19118a0d
# ╠═0d2a3187-46f1-4705-bb3e-ecf74e34840d
# ╠═61513f88-c126-43ab-9515-85a2c8a472bf
# ╠═2c3f3103-9ccb-446e-a289-87bbd0a1b97b
# ╠═795c510e-6280-42d9-8820-290ca07be88d
# ╠═e450329b-46ee-4f53-b6f0-891d6596938c
# ╟─7299bfea-e23e-4993-bebd-9d8762427294
# ╠═d48b9f23-b1b8-43f8-a05d-2234b69bf1f0
# ╟─be693f92-de60-4b43-82b3-ee6cceea310a
# ╠═27b39d7d-fc08-4ccc-aea4-b64f8a4f5726
# ╠═ec723b90-1c8b-4a87-8a86-fe17d6e85f22
# ╠═0d186a1b-eb89-4be8-a237-e5a9608f4a30
# ╠═db3389c0-cb00-404c-81a8-8c4ccdab22f4
# ╠═8b486bd8-25a1-4895-99c8-0541c9e3c8b0
# ╟─20571e1a-9687-4d42-98b3-b3bc0b207b3b
# ╟─5e02b2c0-10cf-4746-840d-7017648f89f8
# ╠═95ab6c83-0757-40b3-a116-6d872a074f69
# ╠═db4f871a-3ca1-4bd3-a2c5-da7d6f599281
# ╠═29360c2a-c47b-48a3-b225-248eb3ec8a42
# ╠═b47754ac-cb17-4ea3-a97d-20371fd341a5
# ╠═bb75e790-a628-4821-b7b1-11cd34a73d38
# ╟─d1312080-272c-4bd5-bcfe-bf47f8320961
# ╠═99b6ab91-a022-449c-988c-0e5c5719c910
# ╠═0d006e4d-72e3-42ab-beb3-e5304a4d9ba9
# ╠═8919faec-753d-41d0-b0a8-c84c025b2919
# ╠═be28f764-a415-46ed-a503-68e8d831f8a8
# ╠═a0c0f48d-e0b0-4f45-8e0e-d9ab5743958b
# ╠═814786be-a760-4023-a614-83e727681075
# ╟─2ba83769-89b4-4ba0-8726-9210f394b4e7
# ╟─885cd51d-895f-4996-a23b-780498b5b810
# ╠═4813e57f-557a-4179-afd2-7925687b3d35
# ╟─ea94bdf9-b0b1-42ad-ba44-aaba73ed1192
# ╠═13487e65-5e48-4a37-9bea-f262dd7b6d56
# ╠═b0cc4665-eb45-48ea-9a33-5acf56d2a283
# ╠═73d638bf-30c1-4694-b3a8-4b29c5e3fa65
# ╠═2141849b-675e-406c-8df4-34b2706507af
# ╠═ac097299-0a31-474c-ab26-a4fb24bb9046
# ╠═8ab0f55d-a393-4a8a-a48c-9ced26033f57
# ╟─4bfc2f7d-a5b0-44c7-8bb6-f1b834c1cc51
# ╠═2188a663-5a85-4ce4-bc8d-20383481e59b
# ╟─ccafd0b9-95aa-4e58-8afc-26cd3ee61cc9
# ╠═81eb8a2d-a3a9-45af-a5a5-b96aefd48712
# ╟─01eacbd4-ef37-4524-aecc-2ef9a1044cf8
# ╠═e52aa672-69a9-419b-a992-e7a3d1364fb6
# ╟─d5565717-239b-41ab-b311-d59b383130ed
# ╠═99a3507b-ca03-429f-acde-e2d1ebb32054
# ╠═d4e9b202-242e-4420-986b-12d2ab57af93
# ╟─7fcccb65-4a7b-4527-97be-d25f481f6eaf
# ╠═18b1c55d-a6b5-44f6-b0b3-50bdb0aa9d96
# ╠═506d408e-dc2b-4e12-b917-286e3f4079a2
# ╠═a7c8cb6a-6e17-4d8f-8958-fe3527c5b8e7
# ╟─e55dfad2-db50-459f-ab54-fa7637fc3638
# ╠═d9304d8d-9a34-46f2-908d-42d5cc5f5c5f
# ╠═a6c2d3a2-326a-41dd-864d-aa3662466222
# ╟─bcff2aa8-2387-44c7-a28f-39cd505a7adf
# ╠═2eb6af9c-9ac7-4b64-93fb-c5a2f037d303
# ╠═79f71f9d-b491-4a2c-85a4-29ae8da4f312
# ╟─53ae4070-8818-4d21-8648-19df9319918a
# ╟─8a70e835-dae6-4727-8204-95c87d5c23da
# ╠═520f280d-c78e-433b-a0a2-8ef05b04f7cc
# ╠═a5671f2f-48ca-4896-9428-147dc671d2b9
# ╠═e01114fa-a847-424a-931f-8f42e623109f
# ╠═e6ee1921-57aa-4194-b2eb-8e32fa4a6c44
# ╟─266bf2ed-d625-47be-8a87-21e12052b27f
# ╠═788903f0-a372-4265-bbf3-5b507a705100
# ╠═337bc837-174f-4ed6-bee1-c9e80dc03b52
# ╟─8f1ec8d7-89ab-4bc5-9eef-2b0a69019854
# ╠═d2d21d41-9a4e-428b-b696-6878901849ba
# ╟─98807f0d-f8a5-4fe0-9983-2b95290347d9
# ╠═96286b65-1a22-4458-a399-46579248cce4
# ╠═86aa821b-a373-4814-953e-535f3a33c002
# ╠═49333dc7-b809-4bb1-b285-20a12d8b9209
# ╟─85098c31-04a1-421c-b3f6-25fec9360588
# ╠═7dca2c78-6faf-450a-8096-1c5b9f7f1295
# ╠═b2622dc4-2fd3-4d80-8efd-a6a985a6b2e9
# ╠═0911a08a-4290-455b-9b26-0bf2862296da
# ╠═86fa378b-815d-4c3d-9121-1338ee54f30f
# ╟─0b094198-cf44-41d7-a8dc-fd8fd0716bb4
# ╠═1f1b384a-6588-45a5-9dd3-6de3face8bfb
# ╠═d82adc20-4c8c-4f2c-9839-d03ad7e7f581
# ╠═8110f306-a7bb-43a2-bb36-6182c59b4b2e
# ╠═1a154bb7-93a3-4973-8908-788db77ac294
# ╠═6b1fb808-e993-4c2b-b81b-6710f8206de7
# ╠═437285d4-ec53-4bb7-9966-fcfb5352e205
# ╠═094bf7ee-7c9b-458c-98dc-521768831654
# ╠═faf69c6f-b179-4ca2-aa22-641e9560e17b
# ╠═f6ce8448-d9ce-4453-9e47-dc6443d50f55
# ╠═9a141034-17cb-4d85-a5a2-4724a38dd269
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
