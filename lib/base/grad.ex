alias :math, as: Math
defmodule Enlso.Base.Grad do
    @moduledoc """
    An implementation of auto grad. See http://mathworld.wolfram.com/Gradient.html
    """

    defp finiteDiff(f, x) do
        @doc """
        finite differencing, see: http://mathworld.wolfram.com/FiniteDifference.html

        ## Parameters

        - f: function handler
        - x: list, the diff point
        """
        p = length(x)
        y = Enum.map(x, f)
        mu = 2 * Math.sqrt(1.0e-12) * (1+norm(x))
        diff = List.duplicate(0.0, p)
        
        for j <- 0..p-1 do
            e_j = List.duplicate(0.0, p)
            e_j|>List.replace_at(j, 1.0)
            diff|>List.replace_at(j, mu |> List.duplicate(p) |> Enum.zip(e_j) |>
                Enum.map(fn(x) -> elem(x, 0)*elem(x, 1) end) |>
                Enum.zip(x) |> Enum.map(fn(x) -> elem(x, 0)+elem(x, 1) end) |> f)
        end
        mu_list = List.duplicate(mu, p)
        g = y |> List.duplicate(p) |> Enum.zip(diff) |> Enum.map(fn(x) -> elem(x, 1)-elem(x, 0) end) |>
            Enum.zip(mu_list) |> Enum.map(fn(x) -> elem(x, 0)/elem(x, 1) end)
    end

    defp centralDiff(f, x) do
        @doc """
        central differencing, see: http://mathworld.wolfram.com/CentralDifference.html

        ## Parameters:

        - f: function handler
        - x: list, the diff point
        """
        p = length(x)
        mu = 2 * Math.sqrt(1.0e-12) * (1+norm(x))
        diff1 = List.duplicate(0.0, p)
        diff2 = List.duplicate(0.0, p)
        
        for j <- 0..p-1 do
            e_j = List.duplicate(0.0, p)
            e_j|>List.replace_at(j, 1.0)
            diff1 |> List.replace_at(j, mu |> List.duplicate(p) |> Enum.zip(e_j) |>
                Enum.map(fn(x) -> elem(x, 0)*elem(x, 1) end) |>
                Enum.zip(x) |> Enum.map(fn(x) -> elem(x, 1)+elem(x, 0) end) |> f)
            diff2 |> List.replace_at(j, mu |> List.duplicate(p) |> Enum.zip(e_j) |>
                Enum.map(fn(x) -> elem(x, 0)*elem(x, 1) end) |>
                Enum.zip(x) |> Enum.map(fn(x) -> elem(x, 1)-elem(x, 0) end) |> f)      
        end
        mu_list = List.duplicate(2*mu, p)
        g = diff1 |> Enum.zip(diff2) |> Enum.map(fn(x) -> elem(x, 0)-elem(x, 1) end) |> Enum.zip(mu_list) |>
            Enum.map(fn(x) -> elem(x, 0)/elem(x, 1) end)
    end

    def norm(x) do
        @doc """
        l2 norm

        ## Parameters:

        - x: List
        """
         Enum.reduce(x, &(&1*&1)) |> Math.sqrt
    end

    def grad!(x, f, type \\ "central") do
        @doc """
        the main part

        ## Parameters:
        
        - x: list, the diff point
        - f: function handler
        - type: string which determines the method to calculate the diff, default is central.

        ## Examples

        iex> f = fn(x) x|>Enum.zip(x)|>Enum.map(fn(x)->elem(x,0)*elem(x,1) end)|>Enum.sum
        iex> Enlso.Base.Grad.grad!(f, [1,2,3], "finite")
        [2.00,4.00,6.00]
        iex> Enlso.Base.Grad.grad!(f, [1,2,3])
        [2.00,4.00,6.00]
        """
        cond do
            String.equivalent?(type, "central") ->
                gradient = centralDiff(f, x)
            String.equivalent?(type, "finite") -> 
                gradient = finiteDiff(f, x)
            _ ->
                raise ArgumentError, message:"type is invalid"
        end
    end

end