var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = MParT","category":"page"},{"location":"#MParT","page":"Home","title":"MParT","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for MParT.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [MParT]","category":"page"},{"location":"#MParT.ComposedMap-Tuple{Vector{<:CxxWrap.StdLib.SharedPtr{<:MParT.ConditionalMapBase}}}","page":"Home","title":"MParT.ComposedMap","text":"`ComposedMap(maps::Vector)`\n\nCreates a ComposedMap from a vector of ConditionalMapBase objects.\n\n\n\n\n\n","category":"method"},{"location":"#MParT.TriangularMap-Tuple{Vector{<:CxxWrap.StdLib.SharedPtr{<:MParT.ConditionalMapBase}}}","page":"Home","title":"MParT.TriangularMap","text":"`TriangularMap(maps::Vector)`\n\nCreates a TriangularMap from a vector of ConditionalMapBase objects\n\n\n\n\n\n","category":"method"},{"location":"#MParT.ATMOptions-Tuple{}","page":"Home","title":"MParT.ATMOptions","text":"`ATMOptions(;kwargs...)`\n\nTakes the fields from MParT's ATMOptions as keyword arguments, and assigns the field value based on a String from the kwarg value, e.g.\n\njulia> using MParT\n\njulia> maxDegrees = MultiIndex(2,3) # limit both dimensions by order 3\n\njulia> ATMOptions(opt_alg=\"LD_SLSQP\", maxDegrees=maxDegrees)\n\n\n\n\n\n","category":"method"},{"location":"#MParT.MapOptions-Tuple{}","page":"Home","title":"MParT.MapOptions","text":"`MapOptions(;kwargs...)`\n\nTakes the fields from MParT's MapOptions as keyword arguments, and assigns the field value based on a String from the kwarg value, e.g.\n\njulia> using MParT\n\njulia> MapOptions(basisType=\"HermiteFunctions\")\n\n\n\n\n\n","category":"method"},{"location":"#MParT.TrainOptions-Tuple{}","page":"Home","title":"MParT.TrainOptions","text":"`TrainOptions(;kwargs...)`\n\nTakes the fields from MParT's TrainOptions as keyword arguments, and assigns the field value based on a String from the kwarg value, e.g.\n\njulia> using MParT\n\njulia> TrainOptions(opt_alg=\"LD_SLSQP\")\n\n\n\n\n\n","category":"method"}]
}