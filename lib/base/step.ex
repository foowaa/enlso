alias :math, as Math
defmodule step do
     @moduledoc """
    An implementation of step choosing 
    """
    def Armijo(grad, f, xk, dk) do
     @doc """
     Armijo strategy
     """   
        beta = 0.5
        sigma = 0.2
        n = 0
        smax = 20
        Armijo_helper(grad, f, xk, dk, n, smax, beta, sigma);
    end

    defp Armijo_helper(grad, f, xk, dk, n, smax, beta, sigma) do
        @doc """
        Armijo helper function for recursion
        """
        yk = beta|>Math.pow(n)|>List.duplicate(length(xk))|>Enum.zip(dk)|>Enum.map(fn(x) -> elem(x, 0)*elem(x, 1) end)|>
             |>Enum.zip(xk)|>Enum.map(fn(x) -> elem(x, 0)+elem(x, 1) end)
        
    end

    def Wolfe(grad, f, xk, dk) do
    @doc """
    Wolfe strategy
    """
    end
end