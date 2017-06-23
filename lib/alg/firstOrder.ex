import enlso.base.grad
alias enlso.base.grad, as Grad
import enlso.base.step
alias enlso.base.step, as Step
import Matrix
defmodule enlso.alg.firstOrder do
    @moduledoc """
    first order optimization: gradient descent, conjugate gradient, BFGS
    """
    def gd(f, x0, nmax, epsilon) do
    @doc """
    naive gradient descet
    """
        grad = Grad.grad(x0, f)
        d = grad|>Enum.map(&(-1*&1))
        result = gd_helper(f, x0, nmax, epsilon, d, grad, n)
    end

    defp gd_helper(f, x0, nmax, epsilon, d, grad, n) do
        if Grad.norm(d) > epsilon && n<nmax do
            alpha = Step.Armijo(grad, f, x0, d)
            x0 = alpha |> List.duplicate(length(d)) |> Enum.zip(d) |>
                Enum.map(fn(x) -> elem(x, 0)*elem(x, 1) end) |>
                Enum.zip(x0) |> Enum.map(fn(x) -> elem(x, 0)+elem(x, 1) end)
            n = n+1
            gd_helper(f, x0, nmax, epsilon, d, grad, n)
        end
        result = {x0, f(x0)}
    end

    def cg(f, x0, nmax, epsilon) do
    @doc """
    conjugate gradient
    """
        grad = Grad.grad(x0, f)
        d = grad|>Enum.map(&(-1*&1))
        n = 0
        result = gd_helper(f, x0, nmax, epsilon, d, grad, n)
    end

    defp cg_helper(f, x0, nmax, epsilon, d, grad, n) do
        if Grad.norm(d) > epsilon && n < nmax do
            alpha = Step.Wolfe(grad, f, x0, d)
            x0 = alpha |> List.duplicate(length(d)) |> Enum.zip(d) |>
                Enum.map(fn(x) -> elem(x, 0)*elem(x, 1) end) |>
                Enum.zip(x0) |> Enum.map(fn(x) -> elem(x, 0)+elem(x, 1) end)
            temp = Grad.grad(x0, f)
            beta = (temp|>Enum.zip(temp)|>Enum.map(fn(x) -> elem(x, 0)*elem(x, 1) end)|>Enum.sum) / 
                   (grad|>Enum.zip(grad)|>Enum.map(fn(x) -> elem(x, 0)*elem(x, 1) end)|>Enum.sum)
            d = beta |> List.duplicate(length(d)) |> Enum.zip(d) |>
                Enum.map(fn(x) -> elem(x, 0)*elem(x, 1) end) |>
                Enum.zip(x0) |> Enum.map(fn(x) -> elem(x, 0)-elem(x, 1) end)
            n = n+1
            grad = temp
            gd_helper(f, x0, nmax, epsilon, d, gard, n)
        end
        result = {x0, f(x0)}
    end

    def bfgs(f, x0, nmax, epsilon) do
    @doc """
    Broyden-Fletcher-Goldfarb-Shanno (BFGS)
    """  
        x = List.duplicate(0, length(x0))
        B = Matrix.indent(length(x0))
        grad = Grad.grad(x0, f)
        d = Matrix.inv(B)
    end
end