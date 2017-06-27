alias :math, as Math
alias :random, as Random
import Enlso.Base.Grad
alias Enlso.Base.Grad, as Grad
defmodule Enlso.Base.Step do
     @moduledoc """
    An implementation of step choosing. See https://en.wikipedia.org/wiki/Wolfe_conditions AND 
                                            https://en.wikipedia.org/wiki/Backtracking_line_search
    """
    def Armijo(grad, f, xk, dk) do
     @doc """
     Armijo strategy
     """   
        beta = 0.5
        sigma = 0.2
        n = 0
        smax = 20
        alpha = Armijo_helper(grad, f, xk, dk, n, smax, beta, sigma);
    end

    defp Armijo_helper(grad, f, xk, dk, n, smax, beta, sigma) do
        @doc """
        Armijo helper function for recursion
        """
        y1 = beta|>Math.pow(n)|>List.duplicate(length(xk))|>Enum.zip(dk)|>Enum.map(fn(x) -> elem(x, 0)*elem(x, 1) end)|>
             |>Enum.zip(xk)|>Enum.map(fn(x) -> elem(x, 0)+elem(x, 1) end)|>f
        y2 = beta|>Math.pow(n) * sigma * grad|>Enum.zip(dk)|>Enum.map(fn(x) -> elem(x, 0)*elem(x, 1) end)|>Enum.sum
        n = n + 1
        if y1 > f(xk)+y2 && n<smax do  
            Armijo_helper(grad, f, xk, dk, n, smax, beta, sigma)
        end
        alpha = Math.pow(beta, n)
    end

    def Wolfe(grad, f, xk, dk) do
    @doc """
    Wolfe strategy
    """
        c1 = 0.1
        c2 = 0.9
        alpha = 1
        a = 0
        b = 1.0e20
        n = 0
        smax = 20
        alpha = Wolfe_helper(grad, f, xk, dk, n, c1, c2, a, b)
    end

    defp Wolfe_helper(grad, f, xk, dk, n, c1, c2, a, b) do
        x1 = alpha |> List.duplicate(length(dk)) |> Enum.zip(dk) |>
                Enum.map(fn(x) -> elem(x, 0)*elem(x, 1) end) |>
                Enum.zip(x) |> Enum.map(fn(x) -> elem(x, 0)+elem(x, 1) end)
        y2 = c1 * alpha * grad|>Enum.zip(dk)|>Enum.map(fn(x) -> elem(x, 0)*elem(x, 1) end)|>Enum.sum
        y3 = c2 * grad|>Enum.zip(dk)|>Enum.map(fn(x) -> elem(x, 0)*elem(x, 1) end)|>Enum.sum
        if f(x1) > f(xk)+y2 && n < smax do
            b = alpha
            alpha = (alpha+a)/2
            n = n+1
            Wolfe_helper(grad, f, xk, dk, n, c1, c2, a, b)
        end 
        if Grad.grad(x1, f)|>Enum.zip(dk)|>Enum.map(fn(x) -> elem(x, 0)*elem(x, 1) end)|>Enum.sum < y3 && n < smax do
            a = alpha
            alpha = min(2*alpha, (b+alpha)/2)
            n = n+1
            Wolfe_helper(grad, f, xk, dk, n, c1, c2, a, b)
        end
        if alpha > 1 do
            # http://www.cultivatehq.com/posts/pseudo-random-number-generator-in-elixir/
            Random.seed(:os.timestamp)
            alpha = 0.01+0.09*Random.uniform
        else
            alpha = alpha
        end
    end
end