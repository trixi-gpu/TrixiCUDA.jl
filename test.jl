a = 10

function outer!(x)
    x = 2
    return x
end

a = outer(x)
println(a)  # Will print 30





