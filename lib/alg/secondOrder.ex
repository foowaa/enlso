import Enlso.Base.Grad
alias Enlso.Base.Grad, as: Grad
import Enlso.Base.Hess
alias Enlso.Base.Hess, as: Hess
import Enlso.Base.Step
alias Enlso.Base.Step, as: Step
alias :math, as: Math
import Matrix

defmodule Enlso.Alg.SecondOrder do
    @moduledoc """
    second order optimization: Newton method
    """

    @doc """
    Newton method. See https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization

    ## Parameters:

    - f: function handler
    - x0: initial point
    - nmax: the max round
    - epsilon: the break condition
    """
    def newton(f, x0, nmax \\ 5000, epsilon \\ 1.0e-5) do
        rho = 0.0
        n = 0
        grad = Grad.grad!(x0, f)
        muk = Grad.norm(grad)|>Math.pow(1+rho)
        hes = Hess.hess!(x0, f)
        d = grad|>List.duplicate(length(grad))|>Matrix.mult(Matrix.transpose(hes)|>Matrix.inv)|>
            Enum.at(0)|>Enum.map(&(-1*&1))
        result = newton_helper(f, x0, nmax, epsilon, d, grad, muk, hes, rho, n)
    end

    defp newton_helper(f, x0, nmax, epsilon, d, grad, muk, hes, rho, n) do
        if Grad.norm(grad) > epsilon && n < nmax do
            grad = Grad.grad!(x0, f)
            muk = Grad.norm(grad)|>Math.pow(1+rho)
            hes = Hess.hess!(x0, f)
            d = grad|>List.duplicate(length(grad))|>Matrix.mult(Matrix.transpose(hes)|>Matrix.inv)|>
                Enum.at(0)|>Enum.map(&(-1*&1))
            alpha = Step.wolfe(grad, f, x0, d)
            x0 = alpha |> List.duplicate(length(d)) |> Enum.zip(d) |>
                Enum.map(fn(x) -> elem(x, 0)*elem(x, 1) end) |>
                Enum.zip(x0) |> Enum.map(fn(x) -> elem(x, 0)+elem(x, 1) end)
            n = n+1
            newton_helper(f, x0, nmax, epsilon, d, grad, muk, hes, rho, n)
        end 
        result = {x0, f.(x0)}
    end
end