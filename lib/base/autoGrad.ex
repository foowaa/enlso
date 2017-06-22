alias :math, as: Math
defmodule elsno.base.grad do

 @moduledoc """
 An implementation of auto grad
 """

    defp finiteDiff(f, x) do
    @doc """
    finite differencing
    """
    p = length(x)
    y = Enum.map(x, f)
    mu = 2 * Math.sqrt(1.0e-12) * (1+norm(x))
    diff = List.duplicate(0.0, p)
    e_j = List.duplicate(1.0, p)
    for j <- 0..p do
        Enum.at(diff, j) = mu |> List.duplicate(p) |> Enum.zip(e_j) |>
            Enum.map(fn(x) -> Enum.at(x, 0)*Enum.at(x, 1)) |>
            Enum.zip(x) |> Enum.map(fn(x) -> Enum.at(x, 0)+Enum.at(x, 1)) |> f
    end
    mu_list = List.duplicate(mu, p)
    g = y |> List.duplicate(p) |> Enum.zip(diff) |> Enum.map(fn(x) -> Enum.at(x, 1)-Enum.at(x, 0)) |>
        Enum.zip(mu_list) |> Enum.map(fn(x) -> Enum.at(x, 0)/Enum.at(x, 1))
    end

    defp centralDiff(f, x) do
    @doc """
    central differencing
    """
    p = length(x)
    mu = 2 * Math.sqrt(1.0e-12) * (1+norm(x))
    diff1 = List.duplicate(0, p)
    diff2 = List.duplicate(0, p)
    e_j = List.duplicate(1.0, p)
    for j <- 1..p do
        Enum.at(diff1, j) = mu |> List.duplicate(p) |> Enum.zip(e_j) |>
            Enum.map(fn(x) -> Enum.at(x, 0)*Enum.at(x, 1)) |>
            Enum.zip(x) |> Enum.map(fn(x) -> Enum.at(x, 1)+Enum.at(x, 0)) |> f
        Enum.at(diff2, j) = mu |> List.duplicate(p) |> Enum.zip(e_j) |>
            Enum.map(fn(x) -> Enum.at(x, 0)*Enum.at(x, 1)) |>
            Enum.zip(x) |> Enum.map(fn(x) -> Enum.at(x, 1)-Enum.at(x, 0)) |> f      
    end
    mu_list = List.duplicate(2*mu, p)
    g = diff1 |> Enum.zip(diff2) |> Enum.map(fn(x) -> Enum.at(x, 0)-Enum.at(x, 1)) |> Enum.zip(mu_list) |>
        Enum.map(fn(x) -> Enum.at(x, 0)/Enum.at(x, 1))
    end

    def norm(x) do
    @doc """
    l2 norm
    """
    Enum.reduce(x, &(&1*&1)) |> Math.sqrt
    end

    def grad(f, x, type \\ "central") do
    @doc """
    the main part
    """
    cond do
        String.equivalent?("central") ->
            centralDiff(f, x)
        String.equivalent?("finite") -> 
            finiteDiff(f, x)
        _ ->
            raise ArgumentError, message:"type is invalid"

    end
    end

end