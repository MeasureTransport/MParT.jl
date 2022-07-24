using MParT
using Documenter

DocMeta.setdocmeta!(MParT, :DocTestSetup, :(using MParT); recursive=true)

makedocs(;
    modules=[MParT],
    authors="MIT UQGroup",
    repo="https://github.com/MeasureTransport/MParT.jl/blob/{commit}{path}#{line}",
    sitename="MParT.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://MeasureTransport.github.io/MParT.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MeasureTransport/MParT.jl",
    devbranch="main",
)
