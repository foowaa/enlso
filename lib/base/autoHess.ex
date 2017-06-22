alias :math, as Math 
import enlso.base.grad 
alias enlso.base.grad, as Grad
defmodule enlso.base.hess do
 @moduledoc """
 An implementation of auto hess
 """
    defp finiteHess(f, x) do
    @doc """
    finite differencing
    """
    p = length(x)
    mu = 2 * Math.sqrt(1.0e-12) * (1+Grad.norm(x))
    
    diff = List.duplicate(0, p) |> List.duplicate(p)
    e_j = List.duplicate(0, p)
    for j <- 1..p do
        Enum.at(e_j, j) = 1.0
        g = mu |> List.duplicate(p) |> Enum.zip(e_j) |>
            Enum.map(fn(x) -> Enum.at(x, 0)*Enum.at(x, 1)) |>
            Enum.zip(x) |> Enum.map(fn(x) -> Enum.at(x, 0)+Enum.at(x, 1)) |> f
        
    end
    end

    defp centralHess(f, x) do
    @doc """
    central differencing
    """    
    end

    def hess(f, x, type \\ "central") do
    @doc """
    the main part
    """    
    end
end