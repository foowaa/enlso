import enlso.base.grad
alias enlso.base.grad, as Grad
import enlso.base.hess
alias enlso.base.hess, as Hess
import enlso.base.step
alias enlso.base.step, as Step
alias :math, as Math
import Matrix

defmodule enlso.alg.secondOrder do
    @moduledoc """
    second order optimization: Newton method
    """
    def newton(f, x0, nmax \\ 5000, epsilon \\ 1.0e-5) do
    @doc """
    Newton method
    """
        rho = 0.0
        n = 0
        grad = Grad.grad(x0, f)
        muk = Grad.norm(grad)|>Math.pow(1+rho)
        hes = Hess.hess!(x0, f)
        d = grad|>List.duplicate(length(grad))|>Matrix.mult(Matrix.transpose(hess)|>Matrix.inv)|>
            Enum.at(0)|>Enum.map(&(-1*&1))
        result = newton_helper(f, x0, nmax, epsilon, d, grad, muk, hes, n)
    end

    defp newton_helper(f, x0, nmax, epsilon, d, grad, muk, hes, rho, n) do
        if Grad.norm(grad) > epsilon && n < nmax do
            grad = Grad.grad(x0, f)
            muk = Grad.norm(grad)|>Math.pow(1+rho)
            hes = Hess.hess!(x0, f)
            d = grad|>List.duplicate(length(grad))|>Matrix.mult(Matrix.transpose(hess)|>Matrix.inv)|>
                Enum.at(0)|>Enum.map(&(-1*&1))
            alpha = Step.Wolfe(grad, f, x0, d)
            x0 = alpha |> List.duplicate(length(d)) |> Enum.zip(d) |>
                Enum.map(fn(x) -> elem(x, 0)*elem(x, 1) end) |>
                Enum.zip(x0) |> Enum.map(fn(x) -> elem(x, 0)+elem(x, 1) end)
        end 
        result = {x0, f(x0)}
    end
end